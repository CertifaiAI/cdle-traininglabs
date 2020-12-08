package ai.certifai.solution.classification;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.conf.layers.*;
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
import org.deeplearning4j.zoo.model.SqueezeNet;
import org.deeplearning4j.zoo.model.VGG16;
import org.deeplearning4j.zoo.model.VGG19;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;

/*
This exercise allows you to build a CNN classifier from a multi-class weather image dataset by using transfer learning.
The dataset consists of 4 classes and the dataset ("WeatherImage") is located in the "resources" folder.
There are various pre-trained image classification models. Thus, you are asked to compare between the pre-trained
models  (i.e. VGG16, VGG19 and SqueezeNet). You could use the same set of hyper-parameters (e.g. updater,
learning rate, etc..).

Dataset source:
Ajayi, Gbeminiyi (2018), “Multi-class Weather Dataset for Image Classification”, Mendeley Data, V1, doi: 10.17632/4drtyfjtfy.1

STEPS:
1. Load VGG16 model from ZooModel. View the summary of the model.
2. Configure the model configuration for layers that are not frozen.
3. Build the neural network configuration by using ComputationGraph.
4. Initialize dataset and create training and testing dataset iterator.
5. Start a dashboard to visualize network training and setup listener to capture useful information during training.
6. Train and evaluate the model.
7. Repeat STEP 1 - STEP 6 for VGG19 and SqueezeNet.
8. Compare the findings between the pre-trained models used.
 */



public class CNN {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(CNN.class);

    private static final int outputNum = 4;
    private static final int seed = 123;
    private static final int trainPerc = 80;
    private static final int batchSize = 16;
    private static final String featureExtractionLayer = "fc2";

    public static void main(String[] args) throws IOException, IllegalAccessException {

        // =================================================================================
        // Weather image classifier built with VGG16 pre-trained model
        // =================================================================================


        /**
        // STEP 1: Load VGG16 model from ZooModel. View the summary.
        ZooModel zooModel = VGG16.builder().build();
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();
        log.info(vgg16.summary());


        // STEP 2: Configure the model configurations for layers that are not frozen by using FineTuneConfiguration
        FineTuneConfiguration fineTuneCOnf = new FineTuneConfiguration.Builder()
                .updater(new Nesterovs(5e-5))
                .seed(seed)
                .build();

        // STEP 3: Build the neural network configuration by using ComputationGraph
        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
                .fineTuneConfiguration(fineTuneCOnf)
                .setFeatureExtractor(featureExtractionLayer)
                .removeVertexKeepConnections("predictions")
                .addLayer("predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(4096).nOut(outputNum)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX).build(),
                        "fc2")
                .build();
        log.info(vgg16Transfer.summary());


        // STEP 4: Initialize dataset and create training and testing dataset iterator
        WeatherDataSetIterator.setup(batchSize, trainPerc);
        DataSetIterator trainIter = WeatherDataSetIterator.trainIterator();
        DataSetIterator testIter = WeatherDataSetIterator.testIterator();

        // STEP 5: Visualize network training using dashboard and set up listener to capture information during training.
         UIServer uiServer = UIServer.getInstance();
         StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
         uiServer.attach(statsStorage);
         vgg16Transfer.setListeners(
         new StatsListener( statsStorage),
         new ScoreIterationListener(1),
         new EvaluativeListener(trainIter, 1, InvocationType.EPOCH_END),
         new EvaluativeListener(testIter, 1, InvocationType.EPOCH_END));

        // STEP 6: Train and evaluate the model
        int iter = 0;
        while(trainIter.hasNext()) {
            vgg16Transfer.fit(trainIter.next());
            Evaluation evalTrain = vgg16Transfer.evaluate(trainIter);
            System.out.print("Training evaluation:");
            log.info(evalTrain.stats());
            if (iter% 10 ==0) {
                log.info("Evaluate model at iter" + iter + "...");
                Evaluation eval = vgg16Transfer.evaluate(testIter);
                log.info(eval.stats());
                testIter.reset();
            }
            iter++;
        }

        log.info("Model build complete");



         **/

        // ==============================================================================
        // Weather image classifier built with VGG19 pre-trained model
        // ==============================================================================

        // REPEAT STEP 1 - STEP 6 for VGG19
        /**
        ZooModel zooModel2 = VGG19.builder().build();
        ComputationGraph vgg19 = (ComputationGraph) zooModel2.initPretrained();
        log.info(vgg19.summary());


        FineTuneConfiguration fineTuneCOnf = new FineTuneConfiguration.Builder()
                .updater(new Nesterovs(5e-5))
                .seed(seed)
                .build();

        ComputationGraph vgg19Transfer = new TransferLearning.GraphBuilder(vgg19)
                .fineTuneConfiguration(fineTuneCOnf)
                .setFeatureExtractor(featureExtractionLayer)
                .removeVertexKeepConnections("predictions")
                .addLayer("predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(4096).nOut(outputNum)
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.SOFTMAX).build(),
                        "fc2")
                .build();
        log.info(vgg19Transfer.summary());


        WeatherDataSetIterator.setup(batchSize, trainPerc);
        DataSetIterator trainIter = WeatherDataSetIterator.trainIterator();
        DataSetIterator testIter = WeatherDataSetIterator.testIterator();

         UIServer uiServer = UIServer.getInstance();
         StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
         uiServer.attach(statsStorage);
         vgg19Transfer.setListeners(
         new StatsListener( statsStorage),
         new ScoreIterationListener(1),
         new EvaluativeListener(trainIter, 1, InvocationType.EPOCH_END),
         new EvaluativeListener(testIter, 1, InvocationType.EPOCH_END));

        int iter = 0;
        while(trainIter.hasNext()) {
            vgg19Transfer.fit(trainIter.next());
            if (iter% 10 ==0) {
                log.info("Evaluate model at iter" + iter + "...");
                Evaluation evalTest = vgg19Transfer.evaluate(testIter);
                log.info(evalTest.stats());
                testIter.reset();
            }
            iter++;
        }
        log.info("Model build complete");

         **/

        // ===========================================================================
        // Weather image classifier built with SqueezeNet pre-trained model
        // ===========================================================================


        ZooModel zooModel3 = SqueezeNet.builder().build();
        ComputationGraph squeezeNet = (ComputationGraph) zooModel3.initPretrained();
        log.info(squeezeNet.summary());


        FineTuneConfiguration fineTuneCOnf = new FineTuneConfiguration.Builder()
                .updater(new Nesterovs(5e-5))
                .seed(seed)
                .build();

        ComputationGraph squeezeNetTransfer = new TransferLearning.GraphBuilder(squeezeNet)
                .fineTuneConfiguration(fineTuneCOnf)
                .setFeatureExtractor("drop9")
                .removeVertexKeepConnections("conv10")
                .removeVertexAndConnections("relu10")
                .removeVertexAndConnections("global_average_pooling2d_5")
                .removeVertexAndConnections("loss")
                .addLayer("conv10",
                        new ConvolutionLayer.Builder(1,1).nIn(512).nOut(outputNum)
                                .build(),
                        "drop9")
                .addLayer("conv10_act", new ActivationLayer(Activation.RELU), "conv10")
                .addLayer("global_avg_pool", new GlobalPoolingLayer(PoolingType.AVG), "conv10_act")
                .addLayer("softmax", new ActivationLayer(Activation.SOFTMAX), "global_avg_pool")
                .addLayer("loss", new LossLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).build(), "softmax")
                .setOutputs("loss")
                .build();
        log.info(squeezeNetTransfer.summary());


        WeatherDataSetIterator.setup(batchSize, trainPerc);
        DataSetIterator trainIter = WeatherDataSetIterator.trainIterator();
        DataSetIterator testIter = WeatherDataSetIterator.testIterator();

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
        uiServer.attach(statsStorage);
        squeezeNetTransfer.setListeners(
                new StatsListener( statsStorage),
                new ScoreIterationListener(1),
                new EvaluativeListener(trainIter, 1, InvocationType.EPOCH_END),
                new EvaluativeListener(testIter, 1, InvocationType.EPOCH_END));

        int iter = 0;
        while(trainIter.hasNext()) {
            squeezeNetTransfer.fit(trainIter.next());

            if (iter% 10 ==0) {
                log.info("Evaluate model at iter" + iter + "...");
                Evaluation evalTest = squeezeNetTransfer.evaluate(testIter);
                log.info(evalTest.stats());
                testIter.reset();
            }
            iter++;
        }


        log.info("Model build complete");


    }


}
