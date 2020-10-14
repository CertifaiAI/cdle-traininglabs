/*
 * Copyright (c) 2019 Skymind AI Bhd.
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

package ai.certifai.training.regression.medicalcostprediction;

import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.common.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

public class MedicalCostPrediction {

    private static Logger log = LoggerFactory.getLogger(MedicalCostPrediction.class);

    public static void main(String[] args) throws IOException, InterruptedException {

        /*
         *  We would be using the Medical Cost Prediction to be our regression example.
         *  This dataset is obtained from https://www.kaggle.com/mirichoi0218/insurance
         *  The description of the attributes:
         *  age: age of primary beneficiary
         *  sex: insurance contractor gender, female, male
         *  bmi: Body mass index
         *  children: Number of children covered by health insurance / Number of dependents
         *  smoker: Smoking
         *  region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
         *  charges: Individual medical costs billed by health insurance
         *
         *
         *  TASK:
         *  ------
         *  1. Load the dataset using record reader
         *  2. Create schema based on the description
         *  3. Filling up the parameters for DataSetIterator for regression
         *  4. Splitting the data into training and test set with the ratio of 7:3
         *  5. Build the neural network with 3 hidden layer
         *  6. Fit your model with training set
         *  7. Evaluate your trained model with the test set
         *
         *  Good luck.
         *
         * */

        int batchSize = 128;
        double lr = 0.0015;
        int input = 9;
        int hidden = 1000;
        int output = 1;
        int nEpoch = 30;
        double reg = 0.0001;
        int seed = 428;

        /*
         *  STEP 1: DATA PREPARATION
         *
         * */

        //  Preparing the data
        File file = new ClassPathResource("medicalCost/insurance.csv").getFile();
            /*
             *
             * ENTER YOUR CODE HERE
             *
             * */

        // Declaring the feature names in schema
//        Schema schema = new Schema.Builder()
//                /*
//                 *
//                 * ENTER YOUR CODE HERE
//                 *
//                 * */
//        System.out.println("Initial Schema: " + schema);

        // Building transform process schema
//        TransformProcess transformProcess = new TransformProcess.Builder(schema)
//                /*
//                 *
//                 * ENTER YOUR CODE HERE
//                 *
//                 * */
//        System.out.println("Final Schema: " + transformProcess.getFinalSchema());

        //  Adding the original data to a list for later transform purpose
//        List<List<Writable>> originalData = new ArrayList<>();
//        while (recordReader.hasNext()) {
//            List<Writable> data = recordReader.next();
//            originalData.add(data);
//        }

        // Transform data into final schema
//        List<List<Writable>> transformedData = LocalTransformExecutor.execute(originalData, transformProcess);

        //  Preparing to split the dataset into training set and test set
//        CollectionRecordReader collectionRecordReader = new CollectionRecordReader(transformedData);
//        DataSetIterator iterator = new RecordReaderDataSetIterator(// ENTER YOUR CODE HERE);

//        DataSet dataSet = iterator.next();
//        dataSet.shuffle();

//        SplitTestAndTrain testAndTrain = // ENTER YOUR CODE HERE

//        DataSet train = // ENTER YOUR CODE HERE
//        DataSet test = // ENTER YOUR CODE HERE

//        INDArray features = train.getFeatures();
//        System.out.println("\nFeature shape: " + features.shapeInfoToString() + "\n");

        //  Assigning dataset iterator for training purpose
//        ViewIterator trainIter = new ViewIterator(train, batchSize);
//        ViewIterator testIter = new ViewIterator(test, batchSize);


        /*
         *  STEP 2: MODEL TRAINING
         *
         * */

        //  Configuring the structure of the model
//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                /*
//                 *
//                 * ENTER YOUR CODE HERE
//                 *
//                 * */
//
//        MultiLayerNetwork model = new MultiLayerNetwork(conf);
//        model.init();

        // Initialize UI server for visualization model performance
        log.info("****************************************** UI SERVER **********************************************");
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
//        model.setListeners(new ScoreIterationListener(10), new StatsListener(statsStorage));

        // Model training - fit trainIter into model and evaluate model with testIter for each of nEpoch
        log.info("\n*************************************** TRAINING **********************************************\n");

        long timeX = System.currentTimeMillis();
        for (int i = 0; i < nEpoch; i++) {
            long time = System.currentTimeMillis();
//            trainIter.reset();
            log.info("Epoch " + i);
//            model.fit(// ENTER YOUR CODE HERE);
            time = System.currentTimeMillis() - time;
            log.info("************************** Done an epoch, TIME TAKEN: " + time + "ms **************************");

            log.info("********************************** VALIDATING *************************************************");
//            RegressionEvaluation evaluation = // ENTER YOUR CODE HERE
//            System.out.println(evaluation.stats());
        }
        long timeY = System.currentTimeMillis();
        log.info("\n******************** TOTAL TIME TAKEN: " + (timeY - timeX) + "ms ******************************\n");

        // Print out target values and predicted values
        log.info("\n*************************************** PREDICTION **********************************************");

//        testIter.reset();

//        INDArray targetLabels = test.getLabels();
//        System.out.println("\nTarget shape: " + targetLabels.shapeInfoToString());

//        INDArray predictions = model.output(testIter);
//        System.out.println("\nPredictions shape: " + predictions.shapeInfoToString() + "\n");

        System.out.println("Target \t\t\t Predicted");


//        for (int i = 0; i < targetLabels.rows(); i++) {
//            System.out.println(targetLabels.getRow(i) + "\t\t" + predictions.getRow(i));
//        }

        // Plot the target values and predicted values
//        PlotUtil.visualizeRegression(targetLabels, predictions);

        // Print out model summary
        log.info("\n************************************* MODEL SUMMARY *******************************************");
//        System.out.println(model.summary());

    }
}

