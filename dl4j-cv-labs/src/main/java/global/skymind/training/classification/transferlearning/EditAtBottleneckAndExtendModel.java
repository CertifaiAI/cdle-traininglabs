package global.skymind.training.classification.transferlearning;

import global.skymind.training.classification.DogBreedDataSetIterator;
import org.datavec.image.transform.*;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class EditAtBottleneckAndExtendModel {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(EditAtBottleneckAndExtendModel.class);

    private static int epochs = 10;
    private static int batchSize = 32;
    private static int seed = 123;
    private static int numClasses = 5;
    private static int trainPerc = 80;

    private static final Random randNumGen = new Random(seed);

    public static void main(String[] args) throws Exception {
        /*
        Initialize image augmentation
        */
        ImageTransform horizontalFlip = new FlipImageTransform(1);
        ImageTransform cropImage = new CropImageTransform(25);
        ImageTransform rotateImage = new RotateImageTransform(randNumGen, 15);
        ImageTransform showImage = new ShowImageTransform("Image", 1000);
        boolean shuffle = false;
        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(
                new Pair<>(horizontalFlip, 0.5),
                new Pair<>(rotateImage, 0.5),
                new Pair<>(cropImage, 0.3)
//                ,new Pair<>(showImage,1.0) //uncomment this to show transform image
        );
        ImageTransform transform = new PipelineImageTransform(pipeline, shuffle);

        /*
        Initialize dataset and create training and testing dataset iterator
        */
        DogBreedDataSetIterator.setup(batchSize, trainPerc, transform);
        DataSetIterator trainIter = DogBreedDataSetIterator.trainIterator();
        DataSetIterator testIter = DogBreedDataSetIterator.testIterator();

        /*
        Using pre-configured model
        */
        // Loading vgg16 from zooModel
        ZooModel zooModel = VGG16.builder().build();
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();
        log.info(vgg16.summary());

        // Override the setting for all layers that are not "frozen".
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .activation(Activation.LEAKYRELU)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-4))
                .dropOut(0.2)
                .seed(seed)
                .build();

        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor("block5_pool") //"block5_pool" and below are frozen
                .nOutReplace("fc2",1024, WeightInit.XAVIER) //modify nOut of the "fc2" vertex
                .removeVertexAndConnections("predictions") //remove the final vertex and it's connections
                .addLayer("fc3",new DenseLayer
                        .Builder().activation(Activation.RELU).nIn(1024).nOut(256).build(),"fc2") //add in a new dense layer
                .addLayer("newpredictions",new OutputLayer
                        .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(256)
                        .nOut(numClasses)
                        .build(),"fc3") //add in a final output dense layer,
                // configurations on a new layer here will be override the finetune confs.
                // For eg. activation function will be softmax not RELU
                .setOutputs("newpredictions") //since we removed the output vertex and it's connections we need to specify outputs for the graph
            .build();
        log.info(vgg16Transfer.summary());

        /*
        Start a dashboard to visualize network training
        Setup listener to capture useful information during training.
        */
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
        uiServer.attach(statsStorage);
        vgg16Transfer.setListeners(
                new StatsListener( statsStorage),
                new ScoreIterationListener(5),
                new EvaluativeListener(trainIter, 1, InvocationType.EPOCH_END),
                new EvaluativeListener(testIter, 1, InvocationType.EPOCH_END)
        );

        /*
        Start training
        */
        vgg16Transfer.fit(trainIter, epochs);

    }
}
