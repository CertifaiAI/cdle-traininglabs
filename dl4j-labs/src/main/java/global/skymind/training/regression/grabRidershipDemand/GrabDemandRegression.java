package global.skymind.training.regression.grabRidershipDemand;/*
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

import global.skymind.Helper;
import global.skymind.solution.regression.grabRidershipDemand.GeohashtoLatLonTransform;
import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
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
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.util.ArchiveUtils;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/*
* In this lab, we work on a ridership demand dataset provided by Grab, where we will build a regression model to predict the demand at a given datetime and location.
* The original dataset contains 4 columns:
*       - geohash6: geohash location
*       - day: the day when the ride happens
*       - timestamp: timestamp of the ride
*       - demand: our prediction target, the demand of the ridership at a partciular time and day at a location
* To complete this lab, you are required to perform some data transformation on the csv data first, and train a model on the first 2.5m rows.
*
*
* TASKS:
* -----
* 1. Run the code to download and unzip the dataset
* 2. Create a schema based on th CSV data
* 3. Perform the following data perparation (transform) steps:
*       a. Extract hour from the "timestamp" column
*       b. Rename the "timestamp" column as "hourOfTheDay" column
*       c. Convert the geohash6 column into latitude and longitude data using the GeohashtoLatLon Transform class
*       d. Remove the original geohash6 column
* . Split the dataset into train and val set (** Use 2.5 million rows for training and the rest for validation)
* . Build and train a neural network to predict the ridership demand
* . Perform model evaluation (regression metrics) on validation set and tune hyperparameter
* . Using your trained model, perform model evaluation on th test set, in GrabDemandRegressionTest.java
*
* */

public class GrabDemandRegression {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(global.skymind.solution.regression.grabRidershipDemand.GrabDemandRegression.class);
    public static final int seed = 12345;
    public static final double learningRate = 0.01;
    public static final int nEpochs = 10;
    public static final int batchSize = 1000;
    public static final int nTrain = 2500000; // Num of training samples to use
    private static String dataDir;
    private static String downloadLink;

    public static void main(String[] args) throws IOException, InterruptedException  {

        //STEP 1: DATA PREPARATION
        downloadLink= Helper.getPropValues("dataset.grab.demand.url");
        dataDir= Paths.get(System.getProperty("user.home"),Helper.getPropValues("dl4j_home.data")).toString();

        File parentDir = new File(Paths.get(dataDir,"grab").toString());
        if(!parentDir.exists()) downloadAndUnzip();

//        File inputFile = new File(Paths.get(dataDir,"grab", "Traffic Management", "train", "train.csv").toString());

                /*
                * COMPLETE THE FOLLOWING 2 LINES OF CODE
                * */
//        CSVRecordReader csvRR = null;
//        csvRR.initialize();

//        Schema inputDataSchema = new Schema.Builder()
                        /*
                         *
                         * ENTER YOUR CODE HERE
                         *
                         * */
//                .build();

//        Pattern REPLACE_PATTERN = Pattern.compile("\\:\\d+");

//        Map<String,String> map = new HashMap<>();
//        map.put(REPLACE_PATTERN.toString(), "");

//        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                        /*
                        *
                        * ENTER YOUR CODE HERE
                        *
                        * */
//                .build();

//        //After executing all of these operations, we have a new and different schema:
//        Schema outputSchema = tp.getFinalSchema();
//        System.out.println(outputSchema);

//        //Process the data:
////        List<List<Writable>> originalData = new ArrayList<>();
//        List<List<Writable>> trainData = new ArrayList<>();
//        List<List<Writable>> valData = new ArrayList<>();
//        int i = 0;
//        while(csvRR.hasNext()){
//            if (i < nTrain) {trainData.add(csvRR.next());}
//            else {valData.add(csvRR.next());}
//            i ++ ;
//        }

//        List<List<Writable>> processedDataTrain = LocalTransformExecutor.execute(trainData, tp);
//        List<List<Writable>> processedDataVal = LocalTransformExecutor.execute(valData, tp);

//        //Create iterator from processedData
//        RecordReader collectionRecordReaderTrain = new CollectionRecordReader(processedDataTrain);
//        RecordReader collectionRecordReaderVal = new CollectionRecordReader(processedDataVal);
//
//        DataSetIterator trainIter = new RecordReaderDataSetIterator(collectionRecordReaderTrain,batchSize,4,4,true);
//        DataSetIterator valIter = new RecordReaderDataSetIterator(collectionRecordReaderVal, processedDataVal.size(),4,4,true);


//         // STEP 2: MODEL TRAINING
//        DataNormalization normalizer = new NormalizerStandardize();
//        normalizer.fit(trainIter);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
//        while (trainIter.hasNext()) {
//            normalizer.transform(trainIter.next());
//        }
//        while (valIter.hasNext()) {
//            normalizer.transform(valIter.next());
//        }
////        normalizer.transform(trainIter);     //Apply normalization to the training data
////        normalizer.transform(valIter);         //Apply normalization to the val data
//
//        //Create the network
                    /*
                    * UPDATE THE FOLLOWING LINES
                    * */
//        int numInput = null;
//        int numOutputs = null;
//        int nHidden = null;
//        MultiLayerNetwork net = null;
                    /*
                     *
                     * ENTER YOUR CODE HERE
                     *
                     * */
//        );
//        net.init();
//        net.setListeners(new ScoreIterationListener(10));
//
//        StatsStorage storage = new InMemoryStatsStorage();
//        UIServer server = UIServer.getInstance();
//        server.attach(storage);
//
//        trainIter.reset();
//        net.fit(trainIter, nEpochs);
//
//        valIter.reset();
//        RegressionEvaluation eval = net.evaluateRegression(valIter);
//        System.out.println(eval.stats());
//
//
//        // STEP 3: SAVE MODEL FOR TESTING
//        // Where to save model
//        File locationToSave = new File(System.getProperty("java.io.tmpdir"), "/trained_regression_model.zip");
//        System.out.println(locationToSave.toString());
//
//        // boolean save Updater
//        boolean saveUpdater = false;
//
//        // ModelSerializer needs modelname, saveUpdater, Location
//        ModelSerializer.writeModel(net,locationToSave,saveUpdater);

    }

    private static void downloadAndUnzip() throws IOException {
        String dataPath = new File(dataDir).getAbsolutePath();
        File zipFile = new File(dataPath, "grab.zip");

        log.info("Downloading the dataset from "+downloadLink+ "...");
        FileUtils.copyURLToFile(new URL(downloadLink), zipFile);

        if(!Helper.getCheckSum(zipFile.getAbsolutePath())
                .equalsIgnoreCase(Helper.getPropValues("dataset.grab.demand.hash"))){
            log.info("Downloaded file is incomplete");
            System.exit(0);
        }

        log.info("Unzipping "+zipFile.getAbsolutePath());
        ArchiveUtils.unzipFileTo(zipFile.getAbsolutePath(), dataPath);
    }
}
