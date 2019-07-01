package global.skymind.training.feedforward;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Simplest Network - Predicting an output value from an input value
 * 0.5 -> 0.8
 *
 * Look for LAB STEP below. Uncomment to proceed.
 * 1. Declare the input and output data in INDArray format
 * 2. Set up the network configuration
 * 3. Set up UI performance monitoring (port http://localhost:9000)
 * 4. Declare MultiLayerNetwork, train the network
 * 5. Tune the network
 */

public class SimplestNetwork
{
    private static Logger log = LoggerFactory.getLogger(SimplestNetwork.class);

    public static void main(String[] args) throws Exception
    {

        int seed = 123; // consistent Random Numbers needed for testing. Initial weights are randomized
        int epochs = 50; // Number of epochs(full passes of the data)
        double learningRate = 0.001; //How fast to adjust weights to minimize error
        int numInputs = 1; // number of input nodes
        int numOutputs = 1; // number of output nodes
        int nHidden = 5; // number of hidden nodes


        /*
		#### LAB STEP 1 #####
		Declare the input and output data in INDArray format
        */
        /*
        INDArray input = Nd4j.create(new float[]{(float) 0.5}, new int[]{1,1});
        INDArray output = Nd4j.create(new float[]{(float) 0.8}, new int[]{1,1});
        */

        /*
		#### LAB STEP 2 #####
		Set up the network configuration
        */
        /*
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, 0.9))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(nHidden)
                        .activation(Activation.TANH)
                        .build())
                .layer(1, new OutputLayer.Builder()
                        .activation(Activation.IDENTITY)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .nIn(nHidden)
                        .nOut(numOutputs)
                        .build())
                .pretrain(false)
                .backprop(true)
                .build();
        */


        /*
		#### LAB STEP 3 #####
		Set up UI performance monitoring (port http://localhost:9000)
        Create a web based UI server to show progress as the network trains
        The Listeners for the model are set here as well
        One listener to pass stats to the UI
        and a Listener to pass progress info to the console
        */
        /*
        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);
        */

        /*
		#### LAB STEP 4 #####
		Declare MultiLayerNetwork, train the network
		*/
        /*
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new StatsListener(storage, 10));
        for(int i = 0; i < epochs; ++i)
        {
            log.info("Epoch: " + i);
            model.fit(input, output);
            INDArray predicted = model.output(input);
            log.info("predicted: " + predicted.toString());
            Thread.sleep(100);
        }
        */

        /*
		#### LAB STEP 5 #####
        Tune the network
        */

    }

}