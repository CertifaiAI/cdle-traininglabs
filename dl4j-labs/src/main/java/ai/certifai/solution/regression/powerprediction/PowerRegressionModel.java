package ai.certifai.solution.regression.powerprediction;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class PowerRegressionModel {
    public static void main(String[] args) throws IOException, InterruptedException {
        final int seed = 12345;
        final double learningRate = 0.001;
        final int nEpochs = 120;
        final int batchSize = 50;
        final int nTrain = 6788; // Num of training samples to use

        String path = new ClassPathResource("/power/power.csv").getFile().getAbsolutePath();
        File file = new File(path);
        RecordReader rr = new CSVRecordReader(1,",");
        rr.initialize(new FileSplit(file));
//      schema of the data
        Schema InputDataSchema = new Schema.Builder()
                .addColumnsDouble("Temperature","Ambient Pressure","Relative Humidity","Exhaust Vacuum","Electrical Output")
                .build();
        System.out.println(InputDataSchema);

        TransformProcess tp = new TransformProcess.Builder(InputDataSchema).build();
        List<List<Writable>> trainData = new ArrayList<>();
        List<List<Writable>> valData = new ArrayList<>();

//      splitting the test and training set
        int i=0;
        while(rr.hasNext()){
            if(i<nTrain) {
                trainData.add(rr.next());
            }else{
                valData.add(rr.next());
            }
            i++;
        }
        RecordReader collectionRecordReaderTrain = new CollectionRecordReader(trainData);
        RecordReader collectionRecordReaderVal = new CollectionRecordReader(valData);

        DataSetIterator trainIter = new RecordReaderDataSetIterator(collectionRecordReaderTrain, batchSize, 4, 4, true);
        DataSetIterator valIter = new RecordReaderDataSetIterator(collectionRecordReaderVal, batchSize, 4, 4, true);

//      NN initialization
        MultiLayerNetwork net = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learningRate))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(4)
                        .nOut(400)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(400)
                        .nOut(200)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nOut(1)
                        .activation(Activation.IDENTITY)
                        .build())
                .build());
        net.init();
        net.setListeners(new ScoreIterationListener(100));

        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);
//        net.setListeners(new StatsListener(storage));

        net.fit(trainIter, nEpochs);
        valIter.reset();
        RegressionEvaluation eval = net.evaluateRegression(valIter);
        System.out.println(eval.stats());
    }
}
