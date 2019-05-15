package global.skymind.solution.recurrent.cnnLSTM;
import net.lingala.zip4j.core.ZipFile;
import net.lingala.zip4j.exception.ZipException;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.File;
import java.io.IOException;
/**
 * This is an extension of Human Activity Classification using CNN-LSTM network [2].
 * The hybrid model improves the accuracy of the classification of activity by ~20%.
 *
 * Convolutional Networks for feature learning:
 * Neural Networks can receive inputs solely on raw signal. However, study shown that applying convolutional
 * layers on top of LSTM helps in improving the performance of model [1]. The convolutional layer act as a filter, able to remove outlier,
 * filtering the data or act as a feature detector to extract out features which are highly correlated to the label (in this case the different types of activities).
 * Thus, applying LSTM to "filtered" data will achieve a better performance.
 *
 * TLDR, Convolutional Layers can learn features in time series and Recurrent Layer (LSTM) can learn the temporal dynamics in time series data.
 *
 * Reference:
 * 1. Palaet et al, 2015. Analysis of CNN-based Speech Recognition System using Raw Speech as Input
 * 2. Ordóñez and Rogen 2016. Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition**/

public class humanActivityClassificationCNNLSTM {

    public static final int batchSize = 64;
    public static final int epoch = 15;
    public static final int numClassLabel = 6;
    private static final int numSkipLines = 0;
    private static final double learningRate = 0.05;

    public static void main(String[] args) throws Exception
    {
        /*
        #### LAB STEP 0 #####
		Unzip all data sets
        */
        unzipAllDataSet();

        /*
		#### LAB STEP 1 #####
		Load data into data iterators
        */

        // Path creation for training set and test set.
        //Training set
        File trainBaseDir = new File(System.getProperty("user.home"), ".deeplearning4j/data/humanactivity/train/");
        File trainFeaturesDir = new File(trainBaseDir, "features");
        File trainLabelsDir = new File(trainBaseDir, "labels");
        //Test set
        File testBaseDir = new File(System.getProperty("user.home"), ".deeplearning4j/data/humanactivity/test/");
        File testFeaturesDir = new File(testBaseDir, "features");
        File testLabelsDir = new File(testBaseDir, "labels");

        // Read all files in the created path using CSVSequenceRecordReader and store them as RecordReader object.
        // Do note that we read all features and labels as well.
        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader(numSkipLines,",");
        trainFeatures.initialize(new NumberedFileInputSplit( trainFeaturesDir.getAbsolutePath()+ "/%d.csv", 0, 7351));
        SequenceRecordReader trainLabels = new CSVSequenceRecordReader(numSkipLines, ",");
        trainLabels.initialize(new NumberedFileInputSplit(trainLabelsDir.getAbsolutePath()+"/%d.csv", 0, 7351));
        //Pass RecordReader into dataset iterator
        //training set
        DataSetIterator train = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, batchSize,numClassLabel,false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        SequenceRecordReader testFeatures = new CSVSequenceRecordReader(numSkipLines,",");
        testFeatures.initialize(new NumberedFileInputSplit( testFeaturesDir.getAbsolutePath()+ "/%d.csv", 0, 2946));
        SequenceRecordReader testLabels = new CSVSequenceRecordReader(numSkipLines, ",");
        testLabels.initialize(new NumberedFileInputSplit(testLabelsDir.getAbsolutePath()+"/%d.csv", 0, 2946));
        //Pass RecordReader into dataset iterator
        //test set
        DataSetIterator test = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, batchSize,numClassLabel,false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        /*
		#### LAB STEP 2 #####
		Build the model
        */
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.NONE)
                .inferenceWorkspaceMode(WorkspaceMode.NONE)
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(learningRate))
                .graphBuilder()
                .addInputs("trainFeatures")
                .setOutputs("predictActivity")
                .addLayer("CNN", new Convolution1DLayer.Builder(1)
                                //input shape: data input channels = 9, [minibatch,inputDepth,height,width]=[64, 9, 128, 1]
                                .nIn(9) //numChannel =9
                                .nOut(32) //number of filters
                                .activation(Activation.RELU)
                                .build(),
                        "trainFeatures")
                .addLayer("LSTM", new LSTM.Builder()
                                .nIn(32) //number of channel =  num of filters
                                .nOut(100)
                                .activation(Activation.TANH)
                                .build(),
                        "CNN")
                .addLayer("predictActivity", new RnnOutputLayer.Builder()
                                .nIn(100)
                                .nOut(numClassLabel)
                                .lossFunction(LossFunctions.LossFunction.MCXENT)
                                .activation(Activation.SOFTMAX)
                                .build(),
                        "LSTM")
                .build();
        /*
		#### LAB STEP 3 #####
		Set listener
        */
        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);

        ComputationGraph model = new ComputationGraph(config);
        model.init();
        model.setListeners(new StatsListener(storage, 10));

        /*
		#### LAB STEP 4 #####
		Train the model
        */
        for (int i=0; i<epoch; i++){
            System.out.println("EPOCH: " + i);
            model.fit(train);
            train.reset();
        }

        /*
		#### LAB STEP 5 #####
		Save the model
        */
        File locationToSave = new File("generated-models/trained_ucihar_CNNLSTMmodel.zip");
        //save updater
        boolean saveUpdater = true;
        ModelSerializer.writeModel(model, locationToSave, saveUpdater);
        System.out.println("\n\nTrain network saved at " + locationToSave);

        /*
		#### LAB STEP 6 #####
		Evaluate the model
        */
        System.out.println("***** Test Evaluation *****");
        Evaluation eval = new Evaluation(numClassLabel);
        test.reset();
        DataSet testDataSet = test.next(1);
        INDArray s = testDataSet.getFeatures();
        System.out.println(s);
        while(test.hasNext())
        {
            testDataSet = test.next();
            INDArray[] predicted = model.output(testDataSet.getFeatures());
            INDArray labels = testDataSet.getLabels();

            eval.evalTimeSeries(labels, predicted[0], testDataSet.getLabelsMaskArray());
        }
        System.out.println(eval.confusionToString());
        System.out.println(eval.stats());

    }

    public static void unzip(String source, String destination){
        try {
            ZipFile zipFile = new ZipFile(source);
            zipFile.extractAll(destination);
        } catch (ZipException e) {
            e.printStackTrace();
        }
    }

    public static void unzipAllDataSet(){
        //unzip training data set
        File resourceDir = new File(System.getProperty("user.home"), ".deeplearning4j/data/humanactivity");
        if (!resourceDir.exists()) resourceDir.mkdirs();

        String zipTrainFilePath = null;
        String zipTestFilePath = null;
        try {
            zipTrainFilePath = new ClassPathResource("humanactivity/train.zip").getFile().toString();
            zipTestFilePath = new ClassPathResource("humanactivity/test.zip").getFile().toString();
        } catch (IOException e) {
            e.printStackTrace();
        }
        File trainFolder = new File(resourceDir+"/train");
        if (!trainFolder.exists()) unzip(zipTrainFilePath, resourceDir.toString());


        File testFolder = new File(resourceDir+"/test");
        if (!testFolder.exists()) unzip(zipTestFilePath, resourceDir.toString());

    }




}
