package ai.certifai.training.classification;

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
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;


import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
/***
 * Dataset
 * https://www.kaggle.com/zhangjuefei/birds-bones-and-living-habits
 *
 * @author BoonKhai Yeoh
 */

public class BirdClassification {
    static int seed = 123;
    static int numInput = 10;
    static int numClass = 6;
    static int epoch = 1000;
    static double splitRatio = 0.8;
    static double learningRate = 1e-2;



    public static void main(String[] args) throws Exception{

        //set filepath
        File dataFile = new ClassPathResource("/birdclassify/bird.csv").getFile();

        //File split
        FileSplit fileSplit = new FileSplit(dataFile);


        //set CSV Record Reader and initialize it
        RecordReader rr = new CSVRecordReader(1,',');
        rr.initialize(fileSplit);

//=========================================================================
        //  Step 1 : Build Schema to prepare the data
//=========================================================================

        Schema sc = new Schema.Builder()
                .build();


//=========================================================================
        //  Step 2 : Build TransformProcess to transform the data
//=========================================================================
        TransformProcess tp = new TransformProcess.Builder(sc)
                .build();


//        Checking the schema
//        Schema outputSchema = tp.getFinalSchema();
//        System.out.println(outputSchema);

        List<List<Writable>> allData = new ArrayList<>();

        while(rr.hasNext()){
            allData.add(rr.next());
        }

//        List<List<Writable>> processData = LocalTransformExecutor.execute(allData, tp);

//========================================================================
        //  Step 3 : Create Iterator ,splitting trainData and testData
//========================================================================

        //Create iterator from process data
//        CollectionRecordReader collectionRR = new CollectionRecordReader(processData);

        //Input batch size , label index , and number of label
//        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(collectionRR, processData.size(),-1,numClass);

//        //Create Iterator and shuffle the dat
//        DataSet fullDataset = dataSetIterator.next();
//        fullDataset.shuffle(seed);
//
//        //Input split ratio
//        SplitTestAndTrain testAndTrain = fullDataset.splitTestAndTrain(splitRatio);
//
          //Get train and test dataset
        DataSet trainData = null;
        DataSet testData = null ;

        //printout size
        System.out.println("Training vector : ");
        System.out.println(Arrays.toString(trainData.getFeatures().shape()));
        System.out.println("Test vector : ");
        System.out.println(Arrays.toString(testData.getFeatures().shape()));

//========================================================================
        //  Step 4 : DataNormalization
//========================================================================

        //Data normalization


//========================================================================
        //  Step 5 : Network Configuration
//========================================================================

        //Get network configuration
        MultiLayerConfiguration config = getConfig(numInput, numClass, learningRate);

        //Define network
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

//========================================================================
        //  Step 6 : Setup UI , listeners
//========================================================================

        //UI-Evaluator
        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);

        //Set model listeners
        model.setListeners(new StatsListener(storage, 10));

//========================================================================
        //  Step 7 : Training
//========================================================================

        //Training
//        Evaluation eval;
//        for(int i=0; i < epoch; i++) {
//            model.fit(trainData);
//            eval = model.evaluate(new ViewIterator(testData, processData.size()));
//            System.out.println("EPOCH: " + i + " Accuracy: " + eval.accuracy());
//        }

//========================================================================
        //  Step 8 : Evaluation
//========================================================================

        //Confusion matrix

        //TrainData
//        Evaluation evalTrain = model.evaluate(new ViewIterator(trainData, processData.size()));
//        System.out.print("Train Data");
//        System.out.println(evalTrain.stats());

        //TestData
//        Evaluation evalTest = model.evaluate();
//        System.out.print("Test Data");
//        System.out.print();



    }
    public static MultiLayerConfiguration getConfig(int numInputs, int numOutputs, double learningRate) {

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .list()
                .build();

        return config;
    }
}
