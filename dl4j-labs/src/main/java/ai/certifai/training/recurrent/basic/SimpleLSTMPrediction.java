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

package ai.certifai.training.recurrent.basic;

import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;

import java.io.IOException;

/**
 * Multivariate Time Series Sequence Prediction using single layer LSTM
 *
 * We can perform the prediction in following steps:
 *  1. Initialize the data
 *      All data are stored in CSV formats, which can be found in "SimpleLSTMPrediction" under "resources" folder.
 *      Data are separated into "feature" and "label" folder respectively which each of them hold 4 files respectively.
 *      Each csv file contains a sequence of time series data as summarized below.
 *
 *      +-------+----------+-------+
 *      | .csv | Sequence | Label |
 *      +-------+----------+-------+
 *      | 0     | [10, 20] | 65    |
 *      |       | [20, 25] |       |
 *      |       | [30, 35] |       |
 *      +-------+----------+-------+
 *      | 1     | [20, 25] | 85    |
 *      |       | [30, 35] |       |
 *      |       | [40, 45] |       |
 *      +-------+----------+-------+
 *      | 2     | [30, 35] | 105   |
 *      |       | [40, 45] |       |
 *      |       | [50, 55] |       |
 *      +-------+----------+-------+
 *      | 3     | [50, 55] | 125   |
 *      |       | [60, 65] |       |
 *      +-------+----------+-------+
 *      Note: we only have 2 sequential data in time series 3, we need to perform preprocessing (masking and padding) on this.
 *      Given the sequence shown in each file we would like to predict the output
 *
 *  2. Setup the LSTM configuration
 *
 *  3. Setup UI server for training
 *
 *  4. Train the model
 *
 *  5. Perform time series predictions
 *
 * This example is inspired by Jason Brownlee from Machine Learning Mastery
 * Src: https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
 *
 */
public class SimpleLSTMPrediction {

    private static int numSkipLines = 0;
    private static int batchSize = 1;
    private static double learningRate = 0.01;
    private static int epoch =2000;

    public static void main(String[] args) throws IOException, InterruptedException {

         /*
		#### LAB STEP 1 #####
		Prepare the data
        */
        //Training set


        // Read all files in the created path using CSVSequenceRecordReader and store them as RecordReader object.
        // Do note that we read all features and labels as well.



        //Pass RecordReader into dataset iterator
        //training set

        //Optional: view data for each batch
//        int j = 0;
//        while (train.hasNext()) {
//            System.out.println("Batch: " + j);
//            System.out.println(train.next());
//            j++;
//        }

        /*
        Take note of the feature shape and the label that we are providing to the model.
        For example in 0.csv, we have:
            Feature:
                [[10,20]
                [20,25]
                [30,35]]
            Label:
                65

          Given the sequence shown in feature, we want our model to produce the output 65.

          Try to answer the following questions:
            1. Given the information above, that is the shape of the feature for 1 time series sequence?
            2. What is the shape of the feature after processing the data using SequenceRecordReaderDataSetIterator
            3. Open 3.csv, it is found that the file only contains 2 time steps, try to find which parts of the code that
               solve this problem by adding masking and padding to the datasets

           You can uncomment the followings to look at the feature and label of dataset.
         */

        //DataSet data = train.next();
        //System.out.println("feature: " + data.getFeatures());
        //System.out.println("feature shape: " + data.getFeatures().shapeInfoToString());
        //System.out.println("label: " + data.getLabels());
        //System.out.println("label shape: " + data.getLabels().shapeInfoToString());
        //System.out.println("numInput: " + train.inputColumns());
        //train.reset();

         /*
		#### LAB STEP 2 #####
		Build the model
        */



        /*
		#### LAB STEP 3 #####
		Set listener
        */
        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);

        //create a model using config, initialize it and set a listener


        /*
		#### LAB STEP 4 #####
		Train the model and print out loss for each epoch
        */


        /*
		#### LAB STEP 5 #####
		Inference
        */

//        INDArray testInput1 = Nd4j.create(new double[][][] {{
//                {   10.0000,   20.0000,   30.0000},
//                {   20.0000,   25.0000,   35.0000}}});
//        System.out.println(model.output(testInput1));


    }
}
