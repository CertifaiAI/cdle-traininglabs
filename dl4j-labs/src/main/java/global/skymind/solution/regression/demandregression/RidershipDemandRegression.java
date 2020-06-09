package global.skymind.solution.regression.demandregression;/*
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
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.util.ArchiveUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

// Features to try
// (latlong.getLat() + 90)*180 + latlong.getLon()

public class RidershipDemandRegression {
    private static final int seed = 12345;
    private static final double learningRate = 0.001;
    private static final int nEpochs = 2;
    private static final int batchSize = 512;
    private static final int nTrain = 3000000; // Num of training samples to use
    private static final Logger log = LoggerFactory.getLogger(RidershipDemandRegression.class);
    private static String dataDir;
    private static String downloadLink;

    public static void main(String[] args) throws IOException, InterruptedException {

        /*
         *  STEP 1: DATA PREPARATION
         *
         * */
        downloadLink = Helper.getPropValues("dataset.ridership.demand.url");
        dataDir = Paths.get(System.getProperty("user.home"), Helper.getPropValues("dl4j_home.data")).toString();

        File parentDir = new File(Paths.get(dataDir, "ridership").toString());
        if (!parentDir.exists()) downloadAndUnzip();

        File inputFile = new File(Paths.get(dataDir, "ridership", "train", "train.csv").toString());

        CSVRecordReader csvRR = new CSVRecordReader(1, ',');
        csvRR.initialize(new FileSplit(inputFile));

        Schema inputDataSchema = new Schema.Builder()
                .addColumnString("geohash6")
                .addColumnInteger("day")
                .addColumnString("timestamp")
                .addColumnFloat("demand")
                .build();


        Map<String, String> map = new HashMap<>();
        map.put("\\:\\d+", "");

        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                .replaceStringTransform("timestamp", map)
                .convertToInteger("timestamp")
                .transform(new GeohashtoLatLonTransform.Builder("geohash6")
                        .addLatDerivedColumn("latitude")
                        .addLonDerivedColumn("longitude").build())
                .removeColumns("geohash6")
                .renameColumn("timestamp", "hour")
                .build();


        //After executing all of these operations, we have a new and different schema:
        Schema outputSchema = tp.getFinalSchema();
        System.out.println(outputSchema);

        //Process the data:
//        List<List<Writable>> originalData = new ArrayList<>();
        List<List<Writable>> trainData = new ArrayList<>();
        List<List<Writable>> valData = new ArrayList<>();
        int i = 0;
        while (csvRR.hasNext()) {
            if (i < nTrain) {
                trainData.add(csvRR.next());
            } else {
                valData.add(csvRR.next());
            }
            i++;
        }

        List<List<Writable>> processedDataTrain = LocalTransformExecutor.execute(trainData, tp);
        List<List<Writable>> processedDataVal = LocalTransformExecutor.execute(valData, tp);

        //Create iterator from processedData
        RecordReader collectionRecordReaderTrain = new CollectionRecordReader(processedDataTrain);
        RecordReader collectionRecordReaderVal = new CollectionRecordReader(processedDataVal);

        DataSetIterator trainIter = new RecordReaderDataSetIterator(collectionRecordReaderTrain, batchSize, 4, 4, true);
        DataSetIterator valIter = new RecordReaderDataSetIterator(collectionRecordReaderVal, batchSize, 4, 4, true);


        /*
         *  STEP 2: MODEL TRAINING
         *
         * */
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainIter);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        trainIter.setPreProcessor(normalizer);
        valIter.setPreProcessor(normalizer);

//        while (trainIter.hasNext()) {
//            normalizer.transform(trainIter.next());
//        }
//        while (valIter.hasNext()) {
//            normalizer.transform(valIter.next());
//        }
//        normalizer.transform(trainIter);     //Apply normalization to the training data
//        normalizer.transform(valIter);         //Apply normalization to the val data

        //Create the network
        int numInput = 4;
        int numOutputs = 1;
        int nHidden = 100;
        MultiLayerNetwork net = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.00001)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learningRate))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInput).nOut(400)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder().nOut(200)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.SIGMOID)
                        .nOut(numOutputs).build())
                .build()
        );
        net.init();
        net.setListeners(new ScoreIterationListener(100));

        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);

        net.fit(trainIter, nEpochs);

        RegressionEvaluation eval = net.evaluateRegression(valIter);
        System.out.println(eval.stats());


        /*
         *  STEP 3: SAVE MODEL FOR TESTING
         *
         * */
        // Where to save model
        File locationToSave = new File(System.getProperty("java.io.tmpdir"), "/trained_regression_model.zip");
        System.out.println(locationToSave.toString());

        // boolean save Updater
        boolean saveUpdater = false;

        // ModelSerializer needs modelname, saveUpdater, Location
        ModelSerializer.writeModel(net, locationToSave, saveUpdater);

    }

    private static void downloadAndUnzip() throws IOException {
        String dataPath = new File(dataDir).getAbsolutePath();
        File zipFile = new File(dataPath, "ridership.zip");

        log.info("Downloading the dataset from " + downloadLink + "...");
        FileUtils.copyURLToFile(new URL(downloadLink), zipFile);

        if (!Helper.getCheckSum(zipFile.getAbsolutePath())
                .equalsIgnoreCase(Helper.getPropValues("dataset.ridership.demand.hash"))) {
            log.info("Downloaded file is incomplete");
            System.exit(0);
        }

        log.info("Unzipping " + zipFile.getAbsolutePath());
        ArchiveUtils.unzipFileTo(zipFile.getAbsolutePath(), dataPath);
    }
}
