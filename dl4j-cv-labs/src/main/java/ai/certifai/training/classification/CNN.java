package ai.certifai.training.classification.transferlearning;

import ai.certifai.solution.classification.WeatherDataSetIterator;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.SqueezeNet;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;


/*
You are asked to build a CNN classifier from a weather image dataset by using transfer learning.
The dataset consists of 4 classes and the dataset ("WeatherImage") is located in the "resources" folder.
There are various pre-trained image classification models. Thus, you are asked to compare between the pre-trained
models  (i.e. VGG16, VGG19 and SqueezeNet). You could use the same set of hyper-parameters (e.g. updater,
learning rate, etc..).

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

    private static final Logger log = org.slf4j.LoggerFactory.getLogger(ai.certifai.solution.classification.CNN.class);

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


        //### START CODE HERE ###
        ZooModel zooModel =

         //### END CODE HERE ####


         // STEP 2: Configure the model configurations for layers that are not frozen by using FineTuneConfiguration.
         FineTuneConfiguration fineTuneCOnf = new FineTuneConfiguration.Builder();
        //### START CODE HERE ###


        //### END CODE HERE


        // STEP 3: Build the neural network configuration by using ComputationGraph

        //### START CODE HERE ###


        //### END CODE HERE ###


        // STEP 4: Initialize dataset and create training and testing dataset iterator
        WeatherDataSetIterator.setup(batchSize, trainPerc);
        //### START CODE HERE ###


        //### END CODE HERE ###


        // STEP 5: Visualize network training using dashboard and set up listener to capture information during training.
        //### START CODE HERE


        // ### END CODE HERE

        // STEP 6: Train and evaluate the model





        log.info("Model build complete");

         **/

        // ==============================================================================
        // Weather image classifier built with VGG19 pre-trained model
        // ** Comment out the lines for VGG16 (section above) while working for VGG19
        // ==============================================================================

        // REPEAT STEP 1 - STEP 6 for VGG19

        /**
         //### START CODE HERE




















         //### END CODE HERE

         log.info("Model build complete");
         **/

        // ==========================================================================================
        // Weather image classifier built with SqueezeNet pre-trained model
        // ** Comment out the lines for VGG16 and VGG19 (sections above) while working for SqueezeNet
        // ===========================================================================================

        // REPEAT STEP 1 - STEP 6 for SqueezeNet

        /**
         ZooModel zooModel3 = SqueezeNet.builder().build();
         // ### START CODE HERE






















         // ### END CODE HERE

         log.info("Model build complete");
         **/

    }
}


// Compare the findings between the pre-trained model used.
