package ai.certifai.training.classification.Multiclass_boon_khai;

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
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
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


//Dataset
//https://www.kaggle.com/zhangjuefei/birds-bones-and-living-habits
public class MulticlassExcercise {
    static int seed =123;
    static int numInput=10;
    static int numClass=6;
    static int epoch=9000;
    static int batch_size=64;
    static double splitratio =0.7;
    static double learningrate=1e-3;


    public static void main(String[] args) throws Exception{

        //set filepath
        File datafile = new ClassPathResource("/birdclassify/bird.csv").getFile();

        //File split
        FileSplit fileSplit = new FileSplit(datafile);


        //set CSV Record Reader and initialize it
        RecordReader rr= new CSVRecordReader(1,',');
        rr.initialize(fileSplit);

        //Build Schema to prepare the data
        Schema sc = new Schema.Builder()
                .addColumnInteger("id")
                .addColumnsFloat("huml","humw","ulnal","ulnaw","feml","femw","tibl","tibw","tarl","tarw")
                .addColumnCategorical("type", Arrays.asList("SW","W","T", "R", "P", "SO"))
                .build();

        TransformProcess tp = new TransformProcess.Builder(sc)
                .removeColumns("id")
                .categoricalToInteger("type")
                .build();

//        Checking the schema
        Schema outputschema=tp.getFinalSchema();
        System.out.println(outputschema);


        List<List<Writable>> alldata = new ArrayList<>();

        while(rr.hasNext()){
            alldata.add(rr.next());
        }

        List<List<Writable>> processdata =LocalTransformExecutor.execute(alldata, tp);

        //Create iterator from process data
        CollectionRecordReader collectionRR = new CollectionRecordReader(processdata);

        //Input batch size , label index , and number of label
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(collectionRR, processdata.size(),10,6);

//        //Create Iterator and shuffle the data
        //Error due to zero data is obtain
        DataSet fulldataset = dataSetIterator.next();
        fulldataset.shuffle();
//
//        //Input split ratio
        SplitTestAndTrain testAndTrain = fulldataset.splitTestAndTrain(splitratio);
//
          //Get train and test dataset
        DataSet trainData =testAndTrain.getTrain();
        DataSet testData =testAndTrain.getTest();

        //printout size
        System.out.println("Training vector : ");
        System.out.println(Arrays.toString(trainData.getFeatures().shape()));
        System.out.println("Test vector : ");
        System.out.println(Arrays.toString(testData.getFeatures().shape()));

        //Data normalization
        DataNormalization normalizer = new NormalizerMinMaxScaler();
        normalizer.fit(trainData);
        normalizer.transform(trainData);
        normalizer.transform(testData);


        //Get network configuration
        MultiLayerConfiguration config = getConfig(numInput, numClass, learningrate);



        //Define network
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.setListeners(new ScoreIterationListener(10));
        model.init();

        //Evaluator
        UIServer server = UIServer.getInstance();
        StatsStorage storage = new InMemoryStatsStorage();
        server.attach(storage);

        //Training
        for (int i = 0; i < epoch; ++i) {
            model.fit(trainData);
        }


        //Confusion matrix
        Evaluation eval = model.evaluate(new ViewIterator(testData, processdata.size()));
        System.out.println(eval.stats());



    }
    public static MultiLayerConfiguration getConfig(int numInputs, int numOutputs, double learningRate) {

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, Nesterovs.DEFAULT_NESTEROV_MOMENTUM))
                .l2(0.001)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(50)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(50)
                        .nOut(100)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nIn(100)
                        .nOut(200)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new DenseLayer.Builder()
                        .nIn(200)
                        .nOut(300)
                        .activation(Activation.RELU)
                        .build())
                .layer(4, new OutputLayer.Builder()
                        .nIn(300)
                        .nOut(numOutputs)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        return config;
    }
}
