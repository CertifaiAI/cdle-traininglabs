package global.skymind.solution.VAE;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;


public class VAECreditAnomaly {

//    private static Logger log = LoggerFactory.getLogger(CSVExample.class);

    public static void main(String[] args) throws  Exception {

        //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        int numLinesToSkip = 1;
        char delimiter = ',';
        RecordReader recordReader_normal = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader_normal.initialize(new FileSplit(new ClassPathResource("/tmp/train_scaled2105.csv").getFile()));

        // Load anomalous data set
        RecordReader recordReader_anomalous = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader_anomalous.initialize(new FileSplit(new ClassPathResource("/tmp/test_scaled2105.csv").getFile()));

        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        int labelIndex = 30;
        int numClasses = 2;
        int minibatchSize = 128;

        DataSetIterator iterator_normal = new RecordReaderDataSetIterator(recordReader_normal,minibatchSize,labelIndex,numClasses);
        DataSet trainData = iterator_normal.next();
        trainData.shuffle();

        // Create iterator for test (anomalous) data
        DataSetIterator iterator_anomalous = new RecordReaderDataSetIterator(recordReader_anomalous,minibatchSize,labelIndex,numClasses);
//        DataSet testData = iterator_anomalous.next();
//        testData.shuffle();


//        System.out.println("BEFORE: " + "\n" + trainData.get(13));
        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
//        DataNormalization normalizer = new NormalizerStandardize();
//        normalizer.fit(iterator_normal);     //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
//        normalizer.transform(trainData);
//        System.out.println("AFTER: " + "\n" + trainData.get(13));

        final int numInputs = 30;
        int outputNum = 2;
        long seed = 8;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(3e-4))
                .weightInit(WeightInit.XAVIER)
                .l2(1e-3)
                .list()
                .layer(0, new VariationalAutoencoder.Builder()
                        .activation(Activation.RELU)
                        .encoderLayerSizes(30, 10, 5)                    //2 encoder layers, each of size 10
                        .decoderLayerSizes(5)                    //2 decoder layers, each of size 10
                        .pzxActivationFunction(Activation.IDENTITY)     //p(z|data) activation function
                        //Bernoulli reconstruction distribution + sigmoid activation - for modelling binary data (or data in range 0 to 1)
                        .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID))
                        .nIn(numInputs)                                   //Input size: 29
                        .nOut(outputNum)                                  //Size of the latent variable space: p(z|x) - 32 values
                        .build())
                .pretrain(true)
                .backprop(false).build();

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        // network.setListeners(new ScoreIterationListener(listenerFreq));
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener( statsStorage),new ScoreIterationListener(1));

        int nEpochs = 50;

        //Fit the data (unsupervised training)
        for( int i=0; i<nEpochs; i++ ){
            model.pretrain(iterator_normal);        //Note use of .pretrain(DataSetIterator) not fit(DataSetIterator) for unsupervised training
            System.out.println("Finished epoch " + (i+1) + " of " + nEpochs);
        }

        //Get the variational autoencoder layer:
        org.deeplearning4j.nn.layers.variational.VariationalAutoencoder vae
                = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) model.getLayer(0);


        Evaluation eval = new Evaluation(2);

        //Iterate over the test (anomalous) data, calculating reconstruction probabilities

        while(recordReader_anomalous.hasNext()){
            DataSet testData = iterator_anomalous.next();
//            normalizer.transform(testData);
            INDArray features = testData.getFeatures();
            INDArray labels = Nd4j.argMax(testData.getLabels(), 1);   //Labels as integer indexes (from one hot), shape [minibatchSize, 1]
            int nRows = features.rows();

            // Get shape of dataset to create array for storing output later
            int shape = testData.asList().size();

            //Calculate the log probability for reconstructions as per An & Cho
            //Higher is better, lower is worse
            int reconstructionNumSamples = 32;
            INDArray reconstructionErrorEachExample = vae.reconstructionLogProbability(features, reconstructionNumSamples);    //Shape: [minibatchSize, 1]
            INDArray predicted = Nd4j.create(shape, 1);

            int threshold = 0;

            for( int j=0; j<nRows; j++){
                INDArray example = features.getRow(j);
                int label = (int)labels.getDouble(j);
                double score = reconstructionErrorEachExample.getDouble(j);

                if (score <= threshold) {
//                    detected_anomalies_count += 1;
                    predicted.putScalar(j,1);
                }
                else {
                    predicted.putScalar(j,0);
                }
            }

            eval.eval(labels, predicted);
        }


        //Print the evaluation statistics
        System.out.println(eval.stats());

    }
}
