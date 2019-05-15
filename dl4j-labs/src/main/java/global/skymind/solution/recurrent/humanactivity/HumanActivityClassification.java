package global.skymind.solution.recurrent.humanactivity;

import net.lingala.zip4j.core.ZipFile;
import net.lingala.zip4j.exception.ZipException;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import java.io.*;


/**
 * The purpose of this lab is to train the model to classify 6 different type of activities using accelerometer data:
 * 1. Walking
 * 2. Walking Upstairs
 * 3. Walking Downstairs
 * 4. Sitting
 * 5. Standing
 * 6. Laying
 *
 * Such signal data are collected by attaching a smartphone on the waist of 30 subjects and the data is labelled manually.
 * You can find out how they did their experiment here: https://www.youtube.com/watch?v=XOEN9W05_4A
 *
 * Data are available at resources/uci_har_dl4j/ and they are splitted into train and test folder. We will use data available in
 * train folder to train our model and test set to test our model. All data for features and labels are stored in respective folder in
 * "train" and "test" folder with train-test-split ratio of 70:30.
 * You can get the data from:
 *          https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
 *                                                  or
 *          https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
 *
 * Do note that RAW data are not available, the downloaded data are preprocessed data with pre-processing steps below:
 * 1. Noise filter on accelerometer and gyroscope
 * 2. Split data into a fix windows of 2.56s or 128 data points with 50% overlap.
 *    Accelerometer data are also splitted into total acceleration, body acceleration and body gyroscope. Each of these feature have
 *    3-axis component ie x, y, z respectively. Thus we have a total of 9 features.
 *
 * The data we use are splitted into multiple CSVs. All CSV files are named by number which represent the sample. For instance, 1.csv means
 * sample 1 and sample 521 means the 521th sample. In training set, we have a total of 7351 samples and 2946 samples for both feature and label .
 * In side each CSV file, we have 128 rows and 9 columns.
 * The number of rows represent time step or the fix windows size that are extracted from accelerometer data.
 * Each column in 9 columns represents different features respectively.
 *
 * Each sample data stored in "feature" has its respective ground truth label stored in "labels".
 * For example: 8.csv file in "train/features", contains feature data which correspond to an activity that can be found in 8.csv from "train/labels".
 * The label (in integer) corresponds to different activities:
 *      1 -> Walking
 *      2 -> Walking Upstairs
 *      3 -> Walking Downstairs
 *      4 -> Sitting
 *      5 -> Standing
 *      6 -> Laying
 *
 * Look for LAB STEP below. Uncomment to proceed.
 *  1. Load the training and testing data
 *  2. Build the model.
 *  3. Set listener
 *  4. Train the model
 *  5. Save the model
 *  6. Evaluate the results
 *
 * @author LohJZ
 *
 *Reference:
 * 1. https://upcommons.upc.edu/handle/2117/20897
 * 2. https://link.springer.com/chapter/10.1007/978-3-642-35395-6_30
 * 3. https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/
**/

public class HumanActivityClassification {
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
        trainFeatures.initialize(new NumberedFileInputSplit( trainFeaturesDir.getAbsolutePath().replace(" ", "%%20")+ "/%d.csv", 0, 7351));
        SequenceRecordReader trainLabels = new CSVSequenceRecordReader(numSkipLines, ",");
        trainLabels.initialize(new NumberedFileInputSplit(trainLabelsDir.getAbsolutePath().replace(" ", "%%20")+"/%d.csv", 0, 7351));
        //Pass RecordReader into dataset iterator
        //training set
        DataSetIterator train = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, batchSize,numClassLabel,false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        SequenceRecordReader testFeatures = new CSVSequenceRecordReader(numSkipLines,",");
        testFeatures.initialize(new NumberedFileInputSplit( testFeaturesDir.getAbsolutePath().replace(" ", "%%20")+ "/%d.csv", 0, 2946));
        SequenceRecordReader testLabels = new CSVSequenceRecordReader(numSkipLines, ",");
        testLabels.initialize(new NumberedFileInputSplit(testLabelsDir.getAbsolutePath().replace(" ", "%%20")+"/%d.csv", 0, 2946));
        //Pass RecordReader into dataset iterator
        //test set
        DataSetIterator test = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, batchSize,numClassLabel,false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        /*
		#### LAB STEP 2 #####
		Build the model
        */
        int numInput = train.inputColumns();
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
                .addLayer("layer0", new LSTM.Builder()
                                .nIn(numInput)
                                .nOut(100)
                                .activation(Activation.TANH)
                                .build(),
                        "trainFeatures")
                .addLayer("predictActivity", new RnnOutputLayer.Builder()
                                .nIn(100)
                                .nOut(numClassLabel)
                                .lossFunction(LossFunctions.LossFunction.MCXENT)
                                .activation(Activation.SOFTMAX)
                                .build(),
                        "layer0")
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
        File locationToSave = new File("generated-models/trained_ucihar_model.zip");
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
