package org.deeplearning4j.solution.recurrent.character;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Random;

/**LSTM English Names Prediction

 Look for LAB STEP below. Uncomment to proceed.
 1. Split input text file -> split input text into minibatch, with each minibatch containing certain length
 2. Configure network setting
 3. Predict data during a fixed interval of training
 */

public class GravesLSTMNamesGeneration {
    public static void main(String[] args) throws Exception {
        int miniBatchSize = 50;                        //Size of mini batch to use when  training
        int exampleLength = 15;                    //Length of each training example sequence to use.
        int tbpttLength = 100;                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
        int numEpochs = 15;                            //Total number of training epochs
        int generateSamplesEveryNMinibatches = 10;  //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
        int nSamplesToGenerate = 2;                    //Number of samples to generate after each training epoch
        int nCharactersToSample = 10;                //Length of each sample to generate
        String generationInitialization = "Ca";        //Optional character initialization; a random character is used if null
        // Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
        // Initialization characters must all be in CharacterIterator.getMinimalCharacterSet() by default
        Random rng = new Random(12345);

        /*
		#### LAB STEP 1 #####
		Get a DataSetIterator that handles vectorization of text
        */
        CharacterIterator iter = getIterator(miniBatchSize, exampleLength);
        int numCharacters = iter.totalOutcomes();

        /*
		#### LAB STEP 2 #####
		Configure network setting
		*/
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(12345)
            .l2(0.0001)
            .weightInit(WeightInit.XAVIER)
            .updater(new Adam(0.01))
            .list()
            .layer(0, new LSTM.Builder().nIn(iter.inputColumns()).nOut(250)
                .activation(Activation.TANH).build())
            .layer(1, new LSTM.Builder().nIn(250).nOut(250)
                .activation(Activation.TANH).build())
            .layer(2, new LSTM.Builder().nIn(250).nOut(200)
                .activation(Activation.TANH).build())
            .layer(3, new LSTM.Builder().nIn(200).nOut(150)
                .activation(Activation.TANH).build())
            .layer(4, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation(Activation.SOFTMAX) //MCXENT + softmax for classification
                .nIn(150).nOut(numCharacters).build())
            .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
            .build();

        if(conf == null)
        {
            System.out.println("Network configuration is null. Abort");
            return;
        }

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));

        //Print the  number of parameters in the network (and for each layer)
        Layer[] layers = net.getLayers();
        long totalNumParams = 0;
        for (int i = 0; i < layers.length; i++) {
            long nParams = layers[i].numParams();
            System.out.println("Number of parameters in layer " + i + ": " + nParams);
            totalNumParams += nParams;
        }
        System.out.println("Total number of network parameters: " + totalNumParams);

        //Do training, and then generate and print samples from network
        int miniBatchNumber = 0;
        for (int i = 0; i < numEpochs; i++)
        {
            System.out.println("Number of Epochs: " + i);


            while (iter.hasNext())
            {
                DataSet ds = iter.next();

                net.fit(ds);

                /*
                #### LAB STEP 3 #####
                Predict data during a fixed interval of training
                */
                if (++miniBatchNumber % generateSamplesEveryNMinibatches == 0)
                {
                    System.out.println("--------------------");
                    System.out.println("Completed " + miniBatchNumber + " minibatches of size " + miniBatchSize + "x" + exampleLength + " characters");
                    System.out.println("Sampling characters from network given initialization \"" + (generationInitialization == null ? "\n" : generationInitialization) + "\n");
                    String[] samples = sampleCharactersFromNetwork(generationInitialization, net, iter, rng, nCharactersToSample, nSamplesToGenerate);

                    for (int j = 0; j < nSamplesToGenerate; j++)
                    {
                        System.out.print("----- Sample " + j + " -----\n\n");
                        String[] name = samples[j].split("\\r?\\n");
                        System.out.println(name[0]);
                        System.out.println();
                    }
                }
            }
            iter.reset();    //Reset iterator for another epoch
        }

        System.out.println("\n\nExample complete");
    }

    /**
     * Read in training data and stores it locally (temp directory). Then set up and return a simple
     * DataSetIterator that does vectorization based on the text.
     *
     * @param miniBatchSize  Number of text segments in each training mini-batch
     * @param sequenceLength Number of characters in each text segment.
     */
    public static CharacterIterator getIterator(int miniBatchSize, int sequenceLength) throws Exception
    {
        File f = new File(new ClassPathResource("dl4j-labs/src/main/resources/text").getPath() + "/EnglishNames.txt");

        if(!f.exists()) throw new IOException("File does not exist: " + f.getAbsolutePath());	//Download problem?

        char[] validCharacters = CharacterIterator.getMinimalCharacterSet();	//Which characters are allowed? Others will be removed
        return new CharacterIterator(f.getAbsolutePath(), Charset.forName("UTF-8"),
            miniBatchSize, sequenceLength, validCharacters, new Random(12345));
    }

    /** Generate a sample from the network, given an (optional, possibly null) initialization. Initialization
     * can be used to 'prime' the RNN with a sequence you want to extend/continue.<br>
     * Note that the initalization is used for all samples
     * @param initialization String, may be null. If null, select a random character as initialization for all samples
     * @param charactersToSample Number of characters to sample from network (excluding initialization)
     * @param net MultiLayerNetwork with one or more LSTM/RNN layers and a softmax output layer
     * @param iter CharacterIterator. Used for going from indexes back to characters
     */
    private static String[] sampleCharactersFromNetwork(String initialization, MultiLayerNetwork net,
                                                        CharacterIterator iter, Random rng, int charactersToSample, int numSamples ){
        //Set up initialization. If no initialization: use a random character
        if( initialization == null ){
            initialization = String.valueOf(iter.getRandomCharacter());
        }

        //Create input for initialization
        INDArray initializationInput = Nd4j.zeros(numSamples, iter.inputColumns(), initialization.length());
        char[] init = initialization.toCharArray();
        for( int i=0; i<init.length; i++ ){
            int idx = iter.convertCharacterToIndex(init[i]);
            for( int j=0; j<numSamples; j++ ){
                initializationInput.putScalar(new int[]{j,idx,i}, 1.0f);
            }
        }

        StringBuilder[] sb = new StringBuilder[numSamples];
        for( int i = 0; i < numSamples; i++ )
        {
            sb[i] = new StringBuilder(initialization);
        }

        //Sample from network (and feed samples back into input) one character at a time (for all samples)
        //Sampling is done in parallel here
        net.rnnClearPreviousState();
        INDArray output = net.rnnTimeStep(initializationInput);
        output = output.tensorAlongDimension((int)output.size(2)-1,1,0);	//Gets the last time step output

        for( int i = 0; i < charactersToSample; i++ )
        {
            //Set up next input (single time step) by sampling from previous output
            INDArray nextInput = Nd4j.zeros(numSamples, iter.inputColumns());

            //Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
            for( int s = 0; s < numSamples; s++ )
            {
                double[] outputProbDistribution = new double[iter.totalOutcomes()];

                for( int j = 0; j < outputProbDistribution.length; j++ )
                {
                    outputProbDistribution[j] = output.getDouble(s, j);
                }

                int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution,rng);

                nextInput.putScalar(new int[]{s,sampledCharacterIdx}, 1.0f);		//Prepare next time step input
                sb[s].append(iter.convertIndexToCharacter(sampledCharacterIdx));	//Add sampled character to StringBuilder (human readable output)
            }

            output = net.rnnTimeStep(nextInput);	//Do one time step of forward pass
        }

        String[] out = new String[numSamples];

        for( int i = 0; i < numSamples; i++ )
        {
            out[i] = sb[i].toString();
        }

        return out;
    }

    /** Given a probability distribution over discrete classes, sample from the distribution
     * and return the generated class index.
     * @param distribution Probability distribution over classes. Must sum to 1.0
     */
    public static int sampleFromDistribution( double[] distribution, Random rng ){
        double d = 0.0;
        double sum = 0.0;
        for( int t=0; t<10; t++ ){
            d = rng.nextDouble();
            sum = 0.0;
            for( int i=0; i<distribution.length; i++ ){
                sum += distribution[i];
                if( d <= sum ) return i;
            }
            //If we haven't found the right index yet, maybe the sum is slightly
            //lower than 1 due to rounding error, so try again.
        }
        //Should be extremely unlikely to happen if distribution is a valid probability distribution
        throw new IllegalArgumentException("Distribution is invalid? d="+d+", sum="+sum);
    }
}
