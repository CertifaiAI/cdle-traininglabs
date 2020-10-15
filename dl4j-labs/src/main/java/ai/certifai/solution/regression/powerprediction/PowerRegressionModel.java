/*
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

package ai.certifai.solution.regression.powerprediction;

import ai.certifai.solution.regression.PlotUtil;
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
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class PowerRegressionModel {

    private static Logger log = LoggerFactory.getLogger(PowerRegressionModel.class);

    public static void main(String[] args) throws IOException, InterruptedException {
        final int seed = 12345;
        final double learningRate = 0.001;
        final int nEpochs = 120;
        final int batchSize = 50;

        String path = new ClassPathResource("/power/power.csv").getFile().getAbsolutePath();
        File file = new File(path);
        CSVRecordReader rr = new CSVRecordReader(1);
        rr.initialize(new FileSplit(file));
//      schema of the data
        Schema InputDataSchema = new Schema.Builder()
                .addColumnsDouble("Temperature","Ambient Pressure","Relative Humidity","Exhaust Vacuum","Electrical Output")
                .build();
        System.out.println("Initial Schema: " + InputDataSchema);

        TransformProcess tp = new TransformProcess.Builder(InputDataSchema).build();
        //  adding the original data to a list for later transform purpose
        List<List<Writable>> originalData = new ArrayList<>();
        while (rr.hasNext()) {
            List<Writable> data = rr.next();
            originalData.add(data);
        }

        // transform data into final schema
        List<List<Writable>> transformedData = LocalTransformExecutor.execute(originalData, tp);

        //  Preparing to split the dataset into training set and test set
        CollectionRecordReader collectionRecordReader = new CollectionRecordReader(transformedData);
        DataSetIterator iterator = new RecordReaderDataSetIterator(collectionRecordReader, transformedData.size(), 4, 4, true);

        DataSet dataSet = iterator.next();
        dataSet.shuffle();

        SplitTestAndTrain testAndTrain = dataSet.splitTestAndTrain(0.7);

        DataSet train = testAndTrain.getTrain();
        DataSet test = testAndTrain.getTest();

        INDArray features = train.getFeatures();
        System.out.println("\nFeature shape: " + features.shapeInfoToString() + "\n");

        //  Assigning dataset iterator for training purpose
        ViewIterator trainIter = new ViewIterator(train, batchSize);
        ViewIterator testIter = new ViewIterator(test, batchSize);
//      NN initialization
        MultiLayerNetwork model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learningRate))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(4)
                        .nOut(400)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(400)
                        .nOut(200)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nOut(1)
                        .activation(Activation.IDENTITY)
                        .build())
                .build());
        model.init();
        log.info("****************************************** UI SERVER **********************************************");
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new ScoreIterationListener(10), new StatsListener(statsStorage));

        // Model training - fit trainIter into model and evaluate model with testIter for each of nEpoch
        log.info("\n*************************************** TRAINING **********************************************\n");

        long timeX = System.currentTimeMillis();
        for (int i = 0; i < nEpochs; i++) {
            long time = System.currentTimeMillis();
            trainIter.reset();
            log.info("Epoch " + i);
            model.fit(trainIter);
            time = System.currentTimeMillis() - time;
            log.info("************************** Done an epoch, TIME TAKEN: " + time + "ms **************************");

            log.info("********************************** VALIDATING *************************************************");
            RegressionEvaluation evaluation = model.evaluateRegression(testIter);
            System.out.println(evaluation.stats());
        }
        long timeY = System.currentTimeMillis();
        log.info("\n******************** TOTAL TIME TAKEN: " + (timeY - timeX) + "ms ******************************\n");

        // Print out target values and predicted values
        log.info("\n*************************************** PREDICTION **********************************************");

        testIter.reset();

        INDArray targetLabels = test.getLabels();
        System.out.println("\nTarget shape: " + targetLabels.shapeInfoToString());

        INDArray predictions = model.output(testIter);
        System.out.println("\nPredictions shape: " + predictions.shapeInfoToString() + "\n");

        System.out.println("Target \t\t\t Predicted");


        for (int i = 0; i < targetLabels.rows(); i++) {
            System.out.println(targetLabels.getRow(i) + "\t\t" + predictions.getRow(i));
        }

        // Plot the target values and predicted values
        PlotUtil.visualizeRegression(targetLabels, predictions);

        // Print out model summary
        log.info("\n************************************* MODEL SUMMARY *******************************************");
        System.out.println(model.summary());
    }
}
