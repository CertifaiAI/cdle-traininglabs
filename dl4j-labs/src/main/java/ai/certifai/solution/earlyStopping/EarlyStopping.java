package ai.certifai.solution.earlyStopping;



import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;

import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Early Stopping
 *
 * Dataset: https://www.kaggle.com/kumargh/pimaindiansdiabetescsv
 * @author boonkhai yeoh
 */
public class EarlyStopping {
    static int seed = 123;
    static int numInput = 8;
    static int numClass = 2;
    static int epoch = 4000;
    static double splitRatio = 0.8;
    static double learningRate = 0.03;

    //weightArray = use weighted loss for imbalanced class
    //600 data point for class 0 , 267 data point for class 1
    //1 means putting more weightage to the class (more emphasis to the class)
    //0.4 means putting less weightage to the class (less emphasis to the class)
    //the sum of weightArray does not need to be 1
    static INDArray weightArray = Nd4j.create(new double[]{0.4 ,1});



    public static void main(String[] args) throws Exception{

        //set filepath
        File dataFile = new ClassPathResource("/earlyStopping/pima-indians-diabetes.csv").getFile();

        //File split
        FileSplit fileSplit = new FileSplit(dataFile);


        //set CSV Record Reader and initialize it
        RecordReader rr = new CSVRecordReader(1,',');
        rr.initialize(fileSplit);

//=========================================================================
        //  Step 1 : Build Schema to prepare the data
//=========================================================================

        //Build Schema to prepare the data
        Schema sc = new Schema.Builder()
                .addColumnsInteger("Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin")
                .addColumnsFloat("BMI","DiabetesPedigreeFunction")
                .addColumnInteger("Age")
                .addColumnCategorical("Class", Arrays.asList("0","1"))
                .build();

//=========================================================================
        //  Step 2 : Build TransformProcess to transform the data
//=========================================================================

        TransformProcess tp = new TransformProcess.Builder(sc)
                .categoricalToInteger("Class")
                .build();

        //Checking the schema
        Schema outputSchema = tp.getFinalSchema();
        System.out.println(outputSchema);


        List<List<Writable>> allData = new ArrayList<>();

        while(rr.hasNext()){
            allData.add(rr.next());
        }

        List<List<Writable>> processData = LocalTransformExecutor.execute(allData, tp);

//========================================================================
        //  Step 3 : Create Iterator ,splitting trainData and testData
//========================================================================

        //Create iterator from processed data
        CollectionRecordReader collectionRR = new CollectionRecordReader(processData);

        //Input batch size , label index , and number of labels
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(collectionRR, processData.size(),8,numClass);

        //Create Iterator and shuffle the data
        DataSet fullDataset = dataSetIterator.next();
        fullDataset.shuffle(seed);

        //Ratio for train-test split
        SplitTestAndTrain testAndTrain = fullDataset.splitTestAndTrain(splitRatio);

        //Get train and test dataset
        DataSet trainData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        //print size of train and test vectors
        System.out.println("Training vector : ");
        System.out.println(Arrays.toString(trainData.getFeatures().shape()));
        System.out.println("Test vector : ");
        System.out.println(Arrays.toString(testData.getFeatures().shape()));

//========================================================================
        //  Step 4 : DataNormalization
//========================================================================

        //Data normalization
        DataNormalization normalizer = new NormalizerMinMaxScaler();
        normalizer.fit(trainData);
        normalizer.transform(trainData);
        normalizer.transform(testData);


        //Create Dataset Iterator
        DataSetIterator trainIterator = new ViewIterator(trainData, trainData.numExamples());
        DataSetIterator testIterator = new ViewIterator(testData, testData.numExamples());

//========================================================================
        //  Step 5 : Network Configuration
//========================================================================

        //Get network configuration for Early Stopping Training
        MultiLayerConfiguration modelConfig = getConfig(numInput, numClass, learningRate);

//========================================================================
        //  Step 6 : Early Stopping Configuration
//========================================================================
        // Early Stopping performs model training for 1 full cycle and identifies the number of epochs tha results in optimal results
        //epochTerminationConditions - termination condition set by user in terms of maximum number of epochs
        //scoreCalculator - which score should be calculated every epoch? [using test set loss here]
        //evaluateEveryNEpochs - the frequency of model evaluation

       EarlyStoppingConfiguration esConfig = new EarlyStoppingConfiguration.Builder()
               .epochTerminationConditions(new MaxEpochsTerminationCondition(epoch))
               .scoreCalculator(new DataSetLossCalculator(testIterator,true))
               .evaluateEveryNEpochs(1)
               .build();

       // Input Early Stopping Configuration , Network Configuration , trainIterator
       EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConfig,modelConfig,trainIterator);

//========================================================================
        //  Step 7 : Training
//========================================================================
        // Perform model training with Early Stopping configuration
        EarlyStoppingResult result = trainer.fit();

        // Instantiate UI server to visualize training process
        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);

        // Re-initialize model to for new training cycle
        MultiLayerConfiguration modelConfig2 = getConfig(numInput, numClass, learningRate);
        MultiLayerNetwork model = new MultiLayerNetwork(modelConfig2);

        //Set model listeners
        model.init();
        model.setListeners(new StatsListener(storage, 1));

        //Set the number of epoch using the best results from the first Early Stopping training to retrain the model
        //result.getBestModelEpoch() - the optimal epoch number
        System.out.println("Retraining model ........ ");

        Evaluation eval;
        for(int i = 0; i < result.getBestModelEpoch(); i++) {
            model.fit(trainData);
            eval = model.evaluate(testIterator);
            System.out.println("EPOCH: " + i + " Accuracy: " + eval.accuracy());
        }


//========================================================================
        //  Step 8 : Evaluation
//========================================================================

        //Confusion matrix
        Evaluation evalTrain = model.evaluate(trainIterator);
        Evaluation evalTest = model.evaluate(testIterator);
        System.out.print("Train Data");
        System.out.println(evalTrain.stats());

        System.out.print("Test Data");
        System.out.println(evalTest.stats());

        //Early Stopping Details
        System.out.println("Termination reason: " + result.getTerminationReason());
        System.out.println("Termination details: " + result.getTerminationDetails());
        System.out.println("Total epochs: " + result.getTotalEpochs());
        System.out.println("Best epoch number: " + result.getBestModelEpoch());
        System.out.println("Score at best epoch: " + result.getBestModelScore());

    }


    public static MultiLayerConfiguration getConfig(int numInputs, int numOutputs, double learningRate) {

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, Nesterovs.DEFAULT_NESTEROV_MOMENTUM))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(10)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(10)
                        .nOut(20)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nIn(20)
                        .nOut(30)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new DenseLayer.Builder()
                        .nIn(30)
                        .nOut(40)
                        .activation(Activation.RELU)
                        .build())
                .layer(4, new OutputLayer.Builder()
                        .nIn(40)
                        .nOut(numOutputs)
                        .lossFunction(new LossMCXENT(weightArray))
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        return config;
    }

}

