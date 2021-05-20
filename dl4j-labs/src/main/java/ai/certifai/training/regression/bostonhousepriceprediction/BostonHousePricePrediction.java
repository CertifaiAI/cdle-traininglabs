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

package ai.certifai.training.regression.bostonhousepriceprediction;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class BostonHousePricePrediction {
    private static final int seed = 123;
    private static final double learningRate = 0.001;
    private static final int nEpochs = 5000;
    private static final int batchSize = 256;

    public static void main(String[] args) throws IOException, InterruptedException {

        /*
         *  We would be using the Boston Housing Dataset for this regression exercise.
         *  This dataset is obtained from https://www.kaggle.com/vikrishnan/boston-house-prices
         *  This dataset consist of 13 features and 1 label, the description are as follow:
         *
         *  CRIM: Per capita crime rate by town
         *  ZN: Proportion of residential land zoned for lots over 25,000 sq. ft
         *  INDUS: Proportion of non-retail business acres per town
         *  CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
         *  NOX: Nitric oxide concentration (parts per 10 million)
         *  RM: Average number of rooms per dwelling
         *  AGE: Proportion of owner-occupied units built prior to 1940
         *  DIS: Weighted distances to five Boston employment centers
         *  RAD: Index of accessibility to radial highways
         *  TAX: Full-value property tax rate per $10,000
         *  PTRATIO: Pupil-teacher ratio by town
         *  B: 1000(Bk — 0.63)², where Bk is the proportion of [people of African American descent] by town
         *  LSTAT: Percentage of lower status of the population
         *  MEDV: Median value of owner-occupied homes in $1000s (target variable)
         *
         *
         *  TASK:
         *  ------
         *  1.  Load the dataset using record reader
         *  2.  Create schema based on the description given above
         *  3.  Filling up the parameters for DataSetIterator for regression
         *  4.  Splitting the data into training and test set with the ratio of 70/30
         *  5.  Build the neural network with 2 hidden layer
         *  6.  Evaluate your trained model with the test set, you should expect your MSE should be around 2.0e+01
         *  Good luck.
         *
         * */

//       //  Preparing the data
//        File dataFile = new ClassPathResource("boston/bostonHousing.csv").getFile();
//        /*
//         *      Use record reader and file split for data loading
//         *      Approximate around 2 lines of codes
//         * */
//
////        Uncomment this to view the data
////        while(recordReader.hasNext()){
////            System.out.println(recordReader.next());
////        }
////        recordReader.reset();
//
//        // Declaring the feature names in schema
//        Schema inputDataSchema =new Schema.Builder()
//                /*
//                 *
//                 * ENTER YOUR CODE HERE
//                 *
//                 * */
//                .build();
//
//        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
//                .build();
//
//        //  adding the original data to a list for later transform purpose
//        List<List<Writable>> originalData = new ArrayList<>();
//        while(recordReader.hasNext()){
//            List<Writable> data = CSVreader.next();
//            originalData.add(data);
//        }
//
//        List<List<Writable>> transformedData = LocalTransformExecutor.execute(originalData,tp);
//
//        //  Printing out the transformed data
//        for (int i = 0; i< transformedData.size();i++){
//            System.out.println(transformedData.get(i));
//        }
//
//        //  Preparing to split the dataset into training and test set
//        CollectionRecordReader crr = new CollectionRecordReader(transformedData);
//        DataSetIterator dataIter = new RecordReaderDataSetIterator(//YOUR CODE HERE);
//
//        DataSet allData = dataIter.next();
//        allData.shuffle();
//
//        SplitTestAndTrain testTrainSplit = // YOUR CODE HERE;
//
//        DataSet trainingSet = // YOUR CODE HERE;
//        DataSet testSet = // YOUR CODE HERE;
//
//        //  Assigning dataset iterator for training purpose
//        ViewIterator trainIter = new ViewIterator(trainingSet, batchSize);
//        ViewIterator testIter = new ViewIterator(testSet, batchSize);
//
//        //  Configuring the structure of the NN
//        MultiLayerConfiguration conf= new NeuralNetConfiguration.Builder()
//                /*
//                 *
//                 * ENTER YOUR CODE HERE
//                 *
//                 * */
//
//
//        MultiLayerNetwork model = new MultiLayerNetwork(conf);
//        model.init();
//        model.setListeners(new ScoreIterationListener(100));
//
//        //  Fitting the model for nEpochs
//        for(int i =0; i<nEpochs;i++){
//            if(i%1000==0){
//                System.out.println("Epoch: " + i);
//            }
//            model.fit(trainIter);
//        }
//
//        //  Evaluating the outcome of our trained model
//        RegressionEvaluation regEval= // YOUR CODE HERE;
//        System.out.println(regEval.stats());
    }
}
