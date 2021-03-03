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

package ai.certifai.training.dl1_datavec.ex4_datatransformation;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

public class Ex3_WomenChessPlayer {

    private static Logger log = LoggerFactory.getLogger(Ex3_WomenChessPlayer.class);

    private static int batchSize = 64;
    private static int label = 13;
    private static int input = 14;
    private static int hidden = 500;
    private static int output = 2;
    private static double lr = 0.0015;
    private static int epoch = 5;

    public static void main(String[] args) throws IOException, InterruptedException {

        /*
         *  We would be using the Top women chess players dataset in the world sorted by their Standard FIDE rating to be our schema building example.
         *  This dataset is obtained from https://www.kaggle.com/vikasojha98/top-women-chess-players
         *  The description of the attributes:
         *  Fide id: fide id of players
         *  Name: name of players
         *  Federation: federation of players
         *  Gender: gender of players, all female since this is women chess players dataset
         *  Year_of_birth: year of birth of players
         *  Title: chess title of players
         *          GM: Grandmaster
         *          IM: International Master
         *          FM: FIDE Master
         *          CM: Candidate Master
         *          WCM: Woman Candidate Master
         *          WFM: Woman FIDE Master
         *          WGM: Woman Grandmaster
         *          WIM: Woman International Master
         *  Standard_Rating: classical game rating of players
         *  Rapid_rating: rapid game rating of players
         *  Blitz_rating: blitz game rating of players
         *  Inactive_flag: flag of inactivity of players (wi - women inactive)
         *
         * */

        // Step 1: Read dataset using CSV Record Reader
        File file = new ClassPathResource("datavec/womenChessPlayer/top_women_chess_players_aug_2020.csv").getFile();
        RecordReader recordReader = new CSVRecordReader(1, ',');
        recordReader.initialize(new FileSplit(file));

        // Step 2: Build up schema for dataset
        Schema schema = new Schema.Builder()
                /*
                 *
                 * ENTER TOUR CODE HERE
                 *
                 */
                .build();
        System.out.println("Initial Schema: " + schema);

        // Step 3: Using transform process for data preprocessing
        TransformProcess transformProcess = new TransformProcess.Builder(schema)
                /*
                 *
                 * TASK:
                 * -----------
                 * 1. remove null values in "year_of_birth" column
                 * 2. add new integer column named "this_year" with all values of 2020 for age calculation
                 * 3. perform math subtraction between "this_year" column and "year_of_birth" column and store in new column named "age"
                 * 4. convert null values into string "Other" in "title" column
                 * 5. convert "title" column into categorical type
                 * 6. convert "title" column from categorical to one hot encoding
                 * 7. convert null values into string "0" in "rapid_rating" column
                 * 8. convert null values into string "0" in "blitz_rating" column
                 * 9. convert "rapid_rating" column from string type to double type
                 * 10. convert "blitz_rating" column from string type to double type
                 * 11. convert null values into string "active" in "state" column
                 * 12. convert string "wi" into string "inactive" in "state" column
                 * 13. convert "state" column into categorical type
                 * 14. convert "state" column from categorical to numerical encoding
                 * 15. remove columns that didn't help in model performance
                 * ------------
                 *
                 *
                 * ENTER YOUR CODE HERE
                 *
                 */
                .build();
        System.out.println("Final Schema: " + transformProcess.getFinalSchema());

        // Step 4: Transform schema
        // Method 1: Using LocalTransformExecutor

        // Method 2: Using TransformProcessRecordReader

        // Step 5: Data preparation
        // Shuffle dataset

        // Split dataset into training set and test set

        // Assigning dataset iterator for training purpose

        // Data normalization

        // Step 6: Model training
        // OPTIONAL: Uncomment below to start model training
        // train(trainIter, testIter, test);

        log.info("********************* END ****************************");
    }

    public static void train(DataSetIterator trainIter, DataSetIterator testIter, DataSet test) {

        // Configuring the architecture of the model
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(lr))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(input)
                        .nOut(hidden)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(hidden)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(hidden)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(hidden)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(output)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        // Initialize UI server for visualization model performance
        log.info("****************************************** UI SERVER **********************************************");
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new ScoreIterationListener(10), new StatsListener(statsStorage));

        // Model training - fit trainIter into model and evaluate model with testIter for each of nEpoch
        log.info("\n*************************************** TRAINING **********************************************\n");

        long timeX = System.currentTimeMillis();
        for (int i = 0; i < epoch; i++) {
            long time = System.currentTimeMillis();
            System.out.println("Epoch" + i + "\n");
            model.fit(trainIter);
            time = System.currentTimeMillis() - time;
            log.info("************************** Done an epoch, TIME TAKEN: " + time + "ms **************************");

            log.info("********************************** VALIDATING *************************************************");
            Evaluation evaluation = model.evaluate(testIter);
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

        // Print out model summary
        log.info("\n*************************************** MODEL SUMMARY *******************************************");
        System.out.println(model.summary());
    }
}
