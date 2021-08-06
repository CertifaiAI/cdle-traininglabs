/*
 * Copyright (c) 2020-2021 CertifAI Sdn. Bhd.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.certifai.solution.recurrent.basic;

import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.custom.ArgMax;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.lang.String;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;


/**
 This example trains a RNN. When trained we only have to put the first
 character of LEARNSTRING to the rnn, AND IT WILL RECITE THE FOLLOWING CHARS

 @author Peter Grossmann

 Look for LAB STEP below. Uncomment to proceed.
 1. Define a sentence to learn
 2. Create a dedicated list of possible chars
 3. Set user interface listeners for network
 4. Initialize test string with first character. Get predicted data from network.
 */
public class BasicRNNExample
{
    /*
	#### LAB STEP 1 #####
	Define a sentence to learn.
	*/

    // Add a special character at the beginning so the RNN learns the complete string and ends with the marker
    private static final String sampleString = "*This example trains a RNN. Look for lab steps below. Uncomment to proceed.";//"*Der Cottbuser Postkutscher putzt den Cottbuser Postkutschkasten."

    //A list of all possible characters
    private static final List<Character> LEARNSTRING_CHARS_LIST = new ArrayList<>();

    public static void main(String[] args)
    {
        double learningRate = 0.001;
        /*
        #### LAB STEP 2 #####
        Create a dedicated list of possible chars in LEARNSTRING_CHARS_LIST.
        */

        char[] LEARNSTRING = sampleString.toCharArray();
        LinkedHashSet<Character> LEARNSTRING_CHARS = new LinkedHashSet<>();
        for(char c : LEARNSTRING)
        {
            LEARNSTRING_CHARS.add(c);

        }
        LEARNSTRING_CHARS_LIST.addAll(LEARNSTRING_CHARS);

        //Neural net configuration
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new RmsProp(learningRate))
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .miniBatch(false)
                .list()
                .layer(0, new LSTM.Builder()
                        .nIn(LEARNSTRING_CHARS_LIST.size())
                        .nOut(150)
                        .activation(Activation.TANH)
                        .build())
                .layer(1, new LSTM.Builder()
                        .nIn(150)
                        .nOut(100)
                        .activation(Activation.TANH)
                        .build())
                .layer(2, new LSTM.Builder()
                        .nIn(100)
                        .nOut(50)
                        .activation(Activation.TANH)
                        .build())
                .layer(3, new RnnOutputLayer.Builder()
                        .nIn(50)
                        .nOut(LEARNSTRING_CHARS_LIST.size())
                        .lossFunction(LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(config);
        network.init();

        /*
        #### LAB STEP 3 #####
        Set user interface listeners for network
        */

        UIServer server = UIServer.getInstance();
        StatsStorage storage = new InMemoryStatsStorage();
        server.attach(storage);

        network.setListeners(new StatsListener(storage, 10));

        // Create input and output arrays: SAMPLE_INDEX, INPUT_NEURON, SEQUENCE_POSITION
        INDArray currentCharArray = Nd4j.zeros(1, LEARNSTRING_CHARS_LIST.size(), LEARNSTRING.length);
        INDArray nextCharArray = Nd4j.zeros(1, LEARNSTRING_CHARS_LIST.size(), LEARNSTRING.length);

        //loop through our sample-sentense
        int iterIndex = 0;

        for(char currentChar : LEARNSTRING)
        {
            //small hack: when currentChar is the last, take the first char as nextChar
            //Not really required. Added to this hack by adding a starter first character
            char nextChar = LEARNSTRING[(iterIndex + 1) % (LEARNSTRING.length)];

            //input neuron for current-char is 1 at iterIndex
            currentCharArray.putScalar(new int[]{0, LEARNSTRING_CHARS_LIST.indexOf(currentChar), iterIndex}, 1);
            nextCharArray.putScalar(new int[]{0, LEARNSTRING_CHARS_LIST.indexOf(nextChar), iterIndex}, 1);

            iterIndex += 1;
        }

        DataSet trainingData = new DataSet(currentCharArray, nextCharArray);


        for(int iterEpoch = 0; iterEpoch < 600; ++iterEpoch)
        {
            System.out.println("Epoch " + iterEpoch);
            System.out.print("String: ");
            //train the data
            network.fit(trainingData);
            //clear current stance fromm the last example
            network.rnnClearPreviousState();

            /*
            #### LAB STEP 4 #####
            Initialize test string with first character. Get predicted data from network.
            */
            //put the first character into the rnn as an initialization
            INDArray testInit = Nd4j.zeros(1,LEARNSTRING_CHARS_LIST.size(),1);
            testInit.putScalar(LEARNSTRING_CHARS_LIST.indexOf(LEARNSTRING[0]), 1);

            //rnn one step -> IMPORTANT: rnnTimeStep() must be called, not output
            //the output shows what the net thinks what should come next
            INDArray output = network.rnnTimeStep(testInit);

            for(char dummy : LEARNSTRING)
            {
                // first process the last output of the network to a concrete
                // neuron, the neuron with the highest output has the highest
                // chance to get chosen
                int sampledCharacterIdx = Nd4j.getExecutioner().exec(new ArgMax(output,1))[0].getInt(0);

                //print the chosen output
                System.out.print(LEARNSTRING_CHARS_LIST.get(sampledCharacterIdx));

                //use the last output as input
                INDArray nextInput =  Nd4j.zeros(1,LEARNSTRING_CHARS_LIST.size(),1);
                nextInput.putScalar(sampledCharacterIdx, 1);
                output = network.rnnTimeStep(nextInput);
            }
            System.out.println("\n");
        }

        System.out.println("BasicRNNExample completed");
    }

}