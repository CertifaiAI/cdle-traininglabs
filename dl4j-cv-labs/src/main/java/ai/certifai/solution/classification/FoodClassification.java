package ai.certifai.solution.classification;

import org.datavec.image.transform.PipelineImageTransform;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.ResNet50;
import org.deeplearning4j.common.resources.DL4JResources;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.IOException;

public class FoodClassification {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(FoodClassification.class);

    protected static final int numClasses = 5;
    protected static final long seed = 12345;

    private static final int batchSize = 8;
    private static final String featureExtractionLayer = "res2a_branch2c";

    public static void main (String[] args) throws IOException {

        new PipelineImageTransform();

        DL4JResources.setBaseDownloadURL("https://dl4jdata.blob.core.windows.net/");

        // UI server setup
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);

        /*
        * LOAD MODEL
        * */
        log.info("\n\nLoading org.deeplearning4j.transferlearning.resnet50...\n\n");
        ZooModel zooModel = ResNet50.builder().build();
        ComputationGraph resnet50 = (ComputationGraph) zooModel.initPretrained();
        log.info(resnet50.summary());

        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .updater(new Nesterovs(5e-4))
                .seed(seed)
                .build();

        //Construct a new model with the intended architecture and print summary
        ComputationGraph resnet50Transfer = new TransferLearning.GraphBuilder(resnet50)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor(featureExtractionLayer) //the specified layer and below are "frozen"
                .removeVertexKeepConnections("fc1000") //replace the functionality of the final vertex
                .addLayer("fc1000",
                        new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                .nIn(2048).nOut(numClasses)
                                .weightInit(new NormalDistribution(0,0.2*(2.0/(4096+numClasses))))
                                .activation(Activation.SOFTMAX).build(),
                        "flatten_1")
                .build();
        log.info(resnet50Transfer.summary());

        //Dataset iterators
        DogBreedDataSetIterator.setup(batchSize, 80);
        DataSetIterator trainIter = DogBreedDataSetIterator.trainIterator();
        DataSetIterator testIter = DogBreedDataSetIterator.testIterator();
        System.out.println(testIter.getLabels());

        Evaluation eval;
        eval = resnet50Transfer.evaluate(testIter);
        log.info("Eval stats BEFORE fit.....");
        log.info(eval.stats() + "\n");
        testIter.reset();

        resnet50Transfer.setListeners(new StatsListener(statsStorage),new ScoreIterationListener(1));

        int iter = 0;
        while(trainIter.hasNext()) {
            resnet50Transfer.fit(trainIter.next());
            if (iter % 10 == 0) {
                log.info("Evaluate model at iter "+iter +" ....");
                eval = resnet50Transfer.evaluate(testIter);
                log.info(eval.stats());
                testIter.reset();
            }
            iter++;
        }

        log.info("Model build complete");
    }
}
