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

package ai.certifai.solution.regression.demandregression.BostonHousePricePrediction;

import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;

import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;

import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

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
         *  We would be using the Boston Housing Dataset to be our regression example.
         *  This dataset is obtained from https://www.kaggle.com/vikrishnan/boston-house-prices
         *  The description of the attributes:
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
         * */

        //  Preparing the data
        File dataFile = new ClassPathResource("boston/bostonHousing.csv").getFile();
        CSVRecordReader CSVreader = new CSVRecordReader();
        CSVreader.initialize(new FileSplit(dataFile));

        // Declaring the feature names in schema
        Schema inputDataSchema =new Schema.Builder()
                .addColumnDouble("CRIM")
                .addColumnDouble("ZN")
                .addColumnDouble("INDUS")
                .addColumnInteger("CHAS")
                .addColumnDouble("NOX")
                .addColumnDouble("RM")
                .addColumnDouble("AGE")
                .addColumnDouble("DIS")
                .addColumnInteger("RAD")
                .addColumnDouble("TAX")
                .addColumnDouble("PTRATIO")
                .addColumnDouble("B")
                .addColumnDouble("LSTAT")
                .addColumnDouble("MEDV")
                .build();

        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                .build();

        //  adding the original data to a list for later transform purpose
        List<List<Writable>> originalData = new ArrayList<>();
        while(CSVreader.hasNext()){
            List<Writable> data = CSVreader.next();
            originalData.add(data);
        }

        List<List<Writable>> transformedData = LocalTransformExecutor.execute(originalData,tp);

        //  Printing out the transformed data
        for (int i = 0; i< transformedData.size();i++){
            System.out.println(transformedData.get(i));
        }

        //  Preparing to split the dataset into training and test set
        CollectionRecordReader crr = new CollectionRecordReader(transformedData);
        DataSetIterator dataIter = new RecordReaderDataSetIterator(crr,transformedData.size(),13,13, true);

        DataSet allData = dataIter.next();
        allData.shuffle();

        SplitTestAndTrain testTrainSplit = allData.splitTestAndTrain(0.7);

        DataSet trainingSet = testTrainSplit.getTrain();
        DataSet testSet = testTrainSplit.getTest();

        //  Assigning dataset iterator for training purpose
        ViewIterator trainIter = new ViewIterator(trainingSet,batchSize);
        ViewIterator testIter = new ViewIterator(testSet,batchSize);

        //  Configuring the structure of the NN
        MultiLayerConfiguration conf= new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(learningRate))
                .weightInit(WeightInit.XAVIER)
                .l2(0.001)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(13)
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(128)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder()
                        .nIn(64)
                        .nOut(1)
                        .activation(Activation.IDENTITY)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        //  Fitting the model for nEpochs
        for(int i =0; i<nEpochs;i++){
            if(i%1000==0){
                System.out.println("Epoch: " + i);
            }
            model.fit(trainIter);
        }

        //  Evaluating the outcome of our trained model
        RegressionEvaluation regEval= model.evaluateRegression(testIter);
        System.out.println(regEval.stats());
    }
}
