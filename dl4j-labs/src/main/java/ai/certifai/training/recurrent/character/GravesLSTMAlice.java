/*
 *
 *  * ******************************************************************************
 *  *  * Copyright (c) 2019 Skymind AI Bhd.
 *  *  * Copyright (c) 2020 CertifAI Sdn. Bhd.
 *  *  *
 *  *  * This program and the accompanying materials are made available under the
 *  *  * terms of the Apache License, Version 2.0 which is available at
 *  *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *  *
 *  *  * Unless required by applicable law or agreed to in writing, software
 *  *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  *  * License for the specific language governing permissions and limitations
 *  *  * under the License.
 *  *  *
 *  *  * SPDX-License-Identifier: Apache-2.0
 *  *  *****************************************************************************
 *
 *
 */

package ai.certifai.training.recurrent.character;


import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Random;

/**

 GravesLSTM Character modelling example


 Example: Train a LSTM RNN to generates text, one character at a time.
 This example is somewhat inspired by Andrej Karpathy's blog post,
 "The Unreasonable Effectiveness of Recurrent Neural Networks"
 http://karpathy.github.io/2015/05/21/rnn-effectiveness/

 You may find similar examples that train on the wikipedia corpus,
 the google news corpus, the linux kernel source code,
 or the complete works of Shakespeare

 This example uses the story "Alice’s Adventures in Wonderland" by Lewis Carroll as the training data.
 You can get the original txt file from here: https://www.gutenberg.org/files/11/11-0.txt

 The model will learn the story and the words, character by character and try to generate the story provided with a seed of text.

 You could modify this and feed it specific strings to probe what it can do.

 For more details on RNNs in DL4J, see the following:
 http://deeplearning4j.org/usingrnns
 http://deeplearning4j.org/lstm
 http://deeplearning4j.org/recurrentnetwork

 Look for LAB STEP below. Uncomment to proceed.
 1. Split input text file -> split input text into minibatch, with each minibatch containing certain length
 2. Setup character initalization -> to prompt the LSTM with a character sequence to continue/complete
 3. Configure network setting
 4. Set user interface listeners
 5. Train network. Predict, sample data to put in network
 6. Save model
 */

public class GravesLSTMAlice
{

    static final int seedNumber = 12345;

    public static void main(String[] args) throws Exception
    {
        int miniBatchSize = 20;                         //Number of text segments in each training mini-batch
        int exampleLength = 500;                     //Number of characters in each text segment.
        int tbpttLength = 50;                           //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
        double learningRate = 0.005;
        double l2Value = 0.0001;                         //Regularization l2 value - to prevent big weight value initialization
        int lstmLayerSize = 200;                        //Number of units in each GravesLSTM hiddenlayer
        int epochs = 5;                                 //Total number of training epochs
        int generateSamplesEveryNMinibatches = 5;      //How frequently to generate samples from the network?
        int numSamples = 2;					            //Number of samples to generate after each training epoch
        int charactersInEachSample = 500;              //Lenght of each sample to generate

       /*
		#### LAB STEP 1 #####
		Create characters dataset by calling characterIterator
        */

        /*
        CharacterIterator characterIter = getCharacterIterator(miniBatchSize, exampleLength);
        int inputLayerSize = characterIter.inputColumns();
        int outputLayerSize = characterIter.totalOutcomes(); //both are same ( minimal characters length)
        */

        /*
		#### LAB STEP 2 #####
		Setup character initalization -> to prompt the LSTM with a character sequence to continue/complete
        */

        /*
         String generationInitialization = null;		//Optional: random character is used if null
//       String generationInitialization = "CHAPTER ";
        */

        /*
		#### LAB STEP 3 #####
		Configure network setting
		*/

        /*
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seedNumber)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learningRate))
                .l2(l2Value)
                .list()
                .layer(0, new LSTM.Builder()
                        .nIn(inputLayerSize)
                        .nOut(lstmLayerSize)
                        .activation(Activation.TANH)
                        .build())
                .layer(1, new LSTM.Builder()
                        .nIn(lstmLayerSize)
                        .nOut(lstmLayerSize)
                        .activation(Activation.TANH)
                        .build())
                .layer(2, new RnnOutputLayer.Builder()
                        .nIn(lstmLayerSize)
                        .nOut(outputLayerSize)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunction.MCXENT)
                        .build())
                .backpropType(BackpropType.TruncatedBPTT)
                .tBPTTLength(tbpttLength)
                .build();
        */


        /*
        MultiLayerNetwork network = new MultiLayerNetwork(config);
        network.init();
        */

        /*
		#### LAB STEP 4 #####
		Set user interface listeners
		*/

        /*
        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);

        network.setListeners(new StatsListener(storage, 10));
        */

        //Print the  number of parameters in the network (and for each layer)

        /*
        Layer[] layers = network.getLayers();
        long totalNumParams = 0;
        for(int i = 0; i < layers.length; ++i)
        {
            long params = layers[i].numParams();
            System.out.println("Number of parameters in layer " + i + ": " + params);
            totalNumParams += params;
        }
        System.out.println("Total number of network parameters: " + totalNumParams);

        System.out.println("Total number of mini batches: " + (int) ceil(characterIter.getFileCharacters() / (double)(miniBatchSize * exampleLength)));
        */

        /*
		#### LAB STEP 5 #####
		Train network
		Predict, sample data to put in network
        */

        /*
        int miniBatchNumber = 0;
        for(int i = 0; i < epochs; ++i)
        {
            while(characterIter.hasNext())
            {
                System.out.println("Current batch: " + miniBatchNumber++ );

                DataSet dataSet = characterIter.next();
                network.fit(dataSet);

                if( miniBatchNumber % generateSamplesEveryNMinibatches == 0)
                {
                    System.out.println("--------------------");
                    System.out.println("Completed " + miniBatchNumber + " minibatches of size " + miniBatchSize + "x" + exampleLength + " characters" );
                    System.out.println("Sampling characters from network given initialization \"" + (generationInitialization == null ? "" : generationInitialization) + "\"");

                    String[] samples = sampleCharactersFromNetwork(generationInitialization, network, characterIter, new Random(seedNumber), charactersInEachSample, numSamples);

                    //print predicted string
                    System.out.println(samples[samples.length - 1]);

                }
            }

            characterIter.reset(); //Reset iterator for another epoch
        }
        */

        /*
		#### LAB STEP 6#####
		Save model
		*/

        /*
        File locationToSave = new File("dl4j-labs/src/main/resources/text/trained_alice.zip");

        //save updater
        boolean saveUpdater = true;


//      ModelSerializer needs modelname, location, booleanSaveUpdater
        ModelSerializer.writeModel(network, locationToSave, saveUpdater);

        System.out.println("\n\nTrain network saved at " + locationToSave);
        */

    }


    /** Generate a sample from the network, given an (optional, possibly null) initialization. Initialization
     * can be used to 'prime' the RNN with a sequence you want to extend/continue.<br>
     * Note that the initalization is used for all samples
     * @param initString prompt string for initialization, may be null. If null, select a random character as initialization for all samples
     * @param network MultiLayerNetwork with one or more GravesLSTM/RNN layers and a softmax output layer
     * @param characterIter CharacterIterator. Used for going from indexes back to characters
     * @param charactersInEachSample Number of characters to sample from network (excluding initialization)
     * @param numSamples Number of samples to generate
     * @return
     */
    private static String[] sampleCharactersFromNetwork(String initString, MultiLayerNetwork network, CharacterIterator characterIter, Random rng,
                                                        int charactersInEachSample, int numSamples)
    {
        //Set up initialization. If no initialization: use a random character
        if( initString == null )
        {
            initString = String.valueOf(characterIter.getRandomCharacter());
        }

        //Set up input for initialization
        //Input array in this form: numOfSamples, numOfCharacters, initializationStringLength
        INDArray initializationInput = Nd4j.zeros(numSamples, characterIter.inputColumns(), initString.length());

        char[] initChar = initString.toCharArray();

        for( int i = 0; i < initChar.length; ++i)
        {
            int index = characterIter.convertCharacterToIndex(initChar[i]);

            for( int j=0; j < numSamples; ++j)
            {
                initializationInput.putScalar(new int[]{j, index, i}, 1.0f);
            }
        }

        //Sample from network (and feed samples back into input) one character at a time (for all samples)
        //Sampling is done in parallel here
        network.rnnClearPreviousState();
        INDArray outputProb = network.rnnTimeStep(initializationInput);

        //Reduce dimension, gets the last time step output
        //Resulted output has the dimension of [numOfSamples, probabilities of length characterIter.totalOutcomes]
        outputProb = outputProb.tensorAlongDimension((int)outputProb.size(2) - 1, 1, 0);

        //String -> string builder for init string
        StringBuilder[] strBuilder = new StringBuilder[numSamples];
        for( int i = 0; i < numSamples; ++i)
        {
            strBuilder[i] = new StringBuilder(initString);
        }

        for( int i = 0; i < charactersInEachSample; i++)
        {
            //Set up next input (single time step) by sampling from previous output
            INDArray nextInput = Nd4j.zeros(numSamples, characterIter.inputColumns());

            //Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
            for( int s = 0; s < numSamples; ++s)
            {
                double[] outputProbDistribution = new double[ characterIter.totalOutcomes() ];

                for( int j = 0; j < outputProbDistribution.length; ++j )
                {
                    outputProbDistribution[j] = outputProb.getDouble(s, j);
                }

                int sampledCharacterIdx = sampleIndexFromDistribution(outputProbDistribution, rng);

                nextInput.putScalar(new int[]{s, sampledCharacterIdx}, 1.0f);		            //Prepare next time step input
                strBuilder[s].append(characterIter.convertIndexToCharacter(sampledCharacterIdx));	//Add sampled character to StringBuilder (human readable output)
            }

            outputProb = network.rnnTimeStep(nextInput);	//Do one time step of forward pass
        }

        String[] outputString = new String[numSamples];

        for( int i = 0; i < numSamples; ++i)
        {
            outputString[i] = strBuilder[i].toString();
        }

        return outputString;
    }

    /**
     * Given a probability distribution over discrete classes, sample from the distribution
     * and return the generated class index.
     *
     * @param distribution Probability distribution over classes. Must sum to 1.0
     * @param rng random number
     */
    public static int sampleIndexFromDistribution(double[] distribution, Random rng)
    {

        double randomNumber = 0.0;
        double sum = 0.0;

        for(int t = 0; t < 10; ++t)
        {
            randomNumber = rng.nextDouble();
            sum = 0.0;

            for (int i = 0; i < distribution.length; ++i)
            {
                sum += distribution[i];

                if (randomNumber <= sum)
                {
                    return i;
                }
            }
        }

        //Should never happen if distribution is a valid probability distribution
        throw new IllegalArgumentException("Distribution is invalid? randomNumber = " + randomNumber + ", sum = " + sum);
    }

    /**Read from text file, set up and return a simple
     * DataSetIterator that does vectorization based on the text.
     * @param miniBatchSize Number of text segments in each training mini-batch
     * @param sequenceLength Number of characters in each text segment.
     */
    public static CharacterIterator getCharacterIterator(int miniBatchSize, int sequenceLength) throws Exception
    {
        File file = new ClassPathResource("text/alice.txt").getFile();
        String fileLocation = file.getAbsolutePath();

        if(!file.exists()) throw new IOException("File does not exist");

        char[] validCharacters = CharacterIterator.getMinimalCharacterSet();

        return new CharacterIterator(fileLocation, Charset.forName("UTF-8"),
                miniBatchSize, sequenceLength, validCharacters, new Random(seedNumber));
    }


}

