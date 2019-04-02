package global.skymind.solution.recurrent.seatemperature;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.CnnToRnnPreProcessor;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import java.io.*;

/**
 A combinational of CNN and LSTM is used to predict the regional temperature of 8 seas: Bengal, Korean, Black, Mediterranean, Arabian, Japan, Bohai, and Okhotsk Seas from 1981 to 2017.
 Raw data was taken from Earth System Research Laboratory (https://www.esrl.noaa.gov/psd/) and preprocessed into CSV file.
 Each example consists of fifty 2-D temperature grids and every grid is represented by a row in the CSV file.

 Combination of CNN-LSTM is deployed because:
 1. CNN is used since each point in the 2-dimensional grid is related to its neighbor points.
    Furthermore, the data is sequential, and each temperature grid is related to the previous grids.
 2. LSTM can be used to forecast the next-day sea temperature for a given temperature grid.

 For more details please refer to:
 1. https://blog.skymind.ai/convolutional-lstms-for-sea-temperature-forecasting/
 2. https://blog.skymind.ai/convolutional-lstms-for-sea-temperature-forecasting-2/
 3. https://deeplearning4j.org/tutorials/15-sea-temperature-convolutional-lstm-example

 For more information on the convolutional LSTM network structure, see:
 https://www.cv- foundation.org/openaccess/content_cvpr_2015/papers/Ng_Beyond_Short_Snippets_2015_CVPR_paper.pdf
 **/


public class seaTemperaturePrediction {

    public static final int numSkipLines =1;
    public static final int batchSize=32;

    public static final int V_HEIGHT=13;
    public static final int V_WIDTH = 4;
    public static final int kernelSize = 2;
    public static final int numChannels = 1;

    public static final double learningRate = 0.05;
    public static final int epoch = 200;
    public static final int numClassLabel=10;

    public static void main(String[] args) throws Exception
    {
        /*
		#### LAB STEP 1 #####
		Load the train data and test data
		There are a total of 1736 files.
		Perform 70/30 train-test split
        */

        //create path and point to the location of data
        File baseDir = new ClassPathResource("sea_temp").getFile();
        File featuresDir = new File(baseDir, "features");
        File targetsDir= new File(baseDir, "targets");

        //create record reader for training set from CSV file, trainFeature contains input at time step t and trainTarget contains ground truth label at time step t+1.
        //Our task is to take in input at t and predict value at t+1
        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader(numSkipLines,",");
        trainFeatures.initialize(new NumberedFileInputSplit( featuresDir.getAbsolutePath()+ "/%d.csv", 1, 1215));
        SequenceRecordReader trainTargets = new CSVSequenceRecordReader(numSkipLines, ",");
        trainTargets.initialize(new NumberedFileInputSplit(targetsDir.getAbsolutePath()+"/%d.csv", 1, 1215));

        //Pass RecordReader into data set iterator
        DataSetIterator train = new SequenceRecordReaderDataSetIterator(trainFeatures, trainTargets, batchSize,numClassLabel,true, SequenceRecordReaderDataSetIterator.AlignmentMode.EQUAL_LENGTH);

        //create record reader for test data and place into data set iterator
        SequenceRecordReader testFeatures = new CSVSequenceRecordReader(numSkipLines,",");
        testFeatures.initialize(new NumberedFileInputSplit( featuresDir.getAbsolutePath()+ "/%d.csv", 1216, 1736));
        SequenceRecordReader testTargets = new CSVSequenceRecordReader(numSkipLines, ",");
        testTargets.initialize(new NumberedFileInputSplit(targetsDir.getAbsolutePath()+"/%d.csv", 1216, 1736));
        DataSetIterator test = new SequenceRecordReaderDataSetIterator(testFeatures, testTargets, batchSize,numClassLabel,true, SequenceRecordReaderDataSetIterator.AlignmentMode.EQUAL_LENGTH);

        /*
		#### LAB STEP 2 #####
		Configure the neural network and fit the neural network onto training set.
        */
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new ConvolutionLayer.Builder(kernelSize, kernelSize)
                        .updater(new AdaGrad(learningRate))
                        .nIn(1) //1 channel
                        .nOut(7)
                        .stride(2, 2)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new LSTM.Builder()
                        .activation(Activation.SOFTSIGN)
                        .nIn(84)
                        .nOut(200)
                        .updater(new AdaGrad(learningRate))
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .build())
                .layer(2, new RnnOutputLayer.Builder(LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(200)
                        .updater(new AdaGrad(learningRate))
                        .nOut(52)
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .build())
                .inputPreProcessor(0, new RnnToCnnPreProcessor(V_HEIGHT, V_WIDTH, numChannels))
                .inputPreProcessor(1, new CnnToRnnPreProcessor(6,2,7))
                .build();

        //Create UI Server
        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);

        //initialize the network
        MultiLayerNetwork network = new MultiLayerNetwork(config);
        network.init();
        network.setListeners(new StatsListener(storage, 10));

        //train the network
        for (int i=0; i<epoch; i++){
            System.out.println("EPOCH: " + i);
            network.fit(train);
            train.reset();
        }

        /*
		#### LAB STEP 3 #####
		Save the model
        */
        File locationToSave = new File("dl4j-labs/src/main/resources/sea_temp/trained_seatemp.zip");
        boolean saveUpdater = true;
        ModelSerializer.writeModel(network, locationToSave, saveUpdater);
        System.out.println("\n\nTrain network saved at " + locationToSave);

        /*
		#### LAB STEP 4 #####
		Evaluate the results
        */
        System.out.println("***** Test Evaluation *****");
        RegressionEvaluation eval = network.evaluateRegression(test);
        test.reset();
        System.out.println();
        System.out.println(eval.stats());

    }
}
