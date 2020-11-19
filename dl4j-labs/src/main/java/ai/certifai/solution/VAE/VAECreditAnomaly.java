/*
 * Copyright (c) 2019 Skymind Holdings Bhd.
 * Copyright (c) 2020 CertifAI Sdn. Bhd.
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
 */

package ai.certifai.solution.VAE;

import net.lingala.zip4j.ZipFile;
import net.lingala.zip4j.exception.ZipException;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.nn.conf.layers.variational.GaussianReconstructionDistribution;
import org.nd4j.common.io.ClassPathResource;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.evaluation.classification.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

import java.io.File;
import java.io.IOException;


public class VAECreditAnomaly {

    public static void main(String[] args) throws  Exception {

        /*
        STEP 1:
        Unzip and load datasets
        */

        // Unzip all data sets
        unzipAllDataSet();

        // Path creation for training set and test set.
        File trainBaseDir = new File(System.getProperty("user.home"), ".deeplearning4j/data/creditFraudDetection/train/");
        File testBaseDir = new File(System.getProperty("user.home"), ".deeplearning4j/data/creditFraudDetection/test/");

        FileSplit train = new FileSplit(trainBaseDir);
        FileSplit test = new FileSplit(testBaseDir);

        // First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        int numLinesToSkip = 1;
        char delimiter = ',';
        RecordReader recordReader_normal = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader_normal.initialize(train);

        // Load anomalous data set
        RecordReader recordReader_anomalous = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader_anomalous.initialize(test);

        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        int labelIndex = 30;
        int numClasses = 2;
        int minibatchSize = 128;

        DataSetIterator iterator_normal = new RecordReaderDataSetIterator(recordReader_normal,minibatchSize,labelIndex,numClasses);
        DataSet trainData = iterator_normal.next();
        trainData.shuffle();

        // Create iterator for test (anomalous) data
        DataSetIterator iterator_anomalous = new RecordReaderDataSetIterator(recordReader_anomalous,minibatchSize,labelIndex,numClasses);


        /*
        STEP 2:
        Setting up configuration for the VAE model
        */

        final int numInputs = 30; // number of features
        int outputNum = 2; // size of latent variable z
        long seed = 8;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(5e-2))
                .weightInit(WeightInit.XAVIER)
                .l2(1e-3)
                .list()
                .layer(0, new VariationalAutoencoder.Builder()
                        .activation(Activation.TANH)
                        .encoderLayerSizes(20)                    //1 encoder layer with 20 nodes
                        .decoderLayerSizes(15,5)                    //2 decoder layers with 15 and 5 nodes respectively
                        .pzxActivationFunction(Activation.IDENTITY)     //p(z|data) activation function
                        //Gaussian reconstruction distribution + TANH activation
                        .reconstructionDistribution(new GaussianReconstructionDistribution(Activation.TANH))
                        .nIn(numInputs)                                   //Input size: 29
                        .nOut(outputNum)                                  //Size of the latent variable space: p(z|x) - 2 values
                        .build())
                .build();

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();


        /*
        STEP 3:
        Set up training visualisation server
        */

        // UI server setup
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener( statsStorage),new ScoreIterationListener(1));


        /*
        STEP 4:
        Run training for VAE model
        */

        // training epochs
        int nEpochs = 5;

        //Fit the data (unsupervised training)
        for( int i=0; i<nEpochs; i++ ){
            model.pretrain(iterator_normal); //Note use of .pretrain(DataSetIterator) not fit(DataSetIterator) for unsupervised training
            System.out.println("Finished epoch " + (i+1) + " of " + nEpochs);
        }

        /*
        STEP 5:
        Make inferences on anomalous data and evaluate the trained VAE model
        */

        //Get the variational autoencoder layer:
        org.deeplearning4j.nn.layers.variational.VariationalAutoencoder vae
                = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) model.getLayer(0);

        Evaluation eval = new Evaluation(2);

        //Iterate over the test (anomalous) data, calculating reconstruction probabilities
        while(recordReader_anomalous.hasNext()){
            DataSet testData = iterator_anomalous.next();
            INDArray features = testData.getFeatures();
            INDArray labels = Nd4j.argMax(testData.getLabels(), 1).reshape(testData.getLabels().size(0), 1);   //Labels as integer indexes (from one hot), shape [minibatchSize, 1]
            int nRows = features.rows();

            // Get shape of dataset to create array for storing output later
            int shape = testData.asList().size();

            //Calculate the log probability for reconstructions as per An & Cho
            //Higher is better, lower is worse
            int reconstructionNumSamples = 32;
            INDArray reconstructionErrorEachExample = vae.reconstructionLogProbability(features, reconstructionNumSamples);    //Shape: [minibatchSize, 1]
            INDArray predicted = Nd4j.create(shape, 1);

            // Setting threshold to identify anomalies. If reconstruction prob score <= threshold, the data point is anomalous.
            double threshold = -70.0;

            for( int j=0; j<nRows; j++){
                double score = reconstructionErrorEachExample.getDouble(j);

                if (score <= threshold) {
                    predicted.putScalar(j,1);
                }
                else {
                    predicted.putScalar(j,0);
                }
            }

            eval.eval(labels, predicted);
        }

        //Print the evaluation statistics
        System.out.println(eval.stats());

    }


    public static void unzip(String source, String destination){
        try {
            ZipFile zipFile = new ZipFile(source);
            zipFile.extractAll(destination);
        } catch (ZipException e) {
            e.printStackTrace();
        }
    }

    public static void unzipAllDataSet(){
        //unzip training data set
        File resourceDir = new File(System.getProperty("user.home"), ".deeplearning4j/data/creditFraudDetection");
        System.out.println(resourceDir);

        String zipTrainFilePath = null;
        String zipTestFilePath = null;
        try {
            zipTrainFilePath = new ClassPathResource("creditFraudDetection/train_scaled2105.zip").getFile().toString();
            zipTestFilePath = new ClassPathResource("creditFraudDetection/test_scaled2105.zip").getFile().toString();
        } catch (IOException e) {
            e.printStackTrace();
        }

        File trainFolder = new File(resourceDir+"/train");
        if (!trainFolder.exists()) unzip(zipTrainFilePath, trainFolder.toString());

        File testFolder = new File(resourceDir+"/test");
        if (!testFolder.exists()) unzip(zipTestFilePath, testFolder.toString());
    }


}
