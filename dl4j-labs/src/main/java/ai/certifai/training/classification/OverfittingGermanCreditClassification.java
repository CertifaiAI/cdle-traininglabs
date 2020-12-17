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

package ai.certifai.training.classification;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.IntegerColumnCondition;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.impl.LossBinaryXENT;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import static ai.certifai.training.classification.PlotUtil.plotLossGraph;

/***
 * Dataset: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
 *
 * ******************************************************
 *
 * This is an example of overfitting.
 * Overfitting happens when it memorises the training data instead of learning the relationship between features and labels
 * Overfitting can be detected by observing the training/validation accuracy and loss.
 * When in doubt, overfitting is best judged by looking at the loss
 * Signs of overfitting:
 *  - training loss(score) decreases, validation loss(score) increases
 *  - training accuracy increases, validation accuracy decreases
 *
 * Search for the keyword "SOLUTION" to see what can be used to avoid overfitting
 *
 * ******************************************************
 */

public class OverfittingGermanCreditClassification {

    private static int labelIndex = 63; // index of the column of the labels/classes
    private static int numOfClasses = 2; // number of classification labels/classes
    private static int numOfFeatures = 63; // number of features to be fed to the model
    private static int numOfEpochs = 10000; // number of Epochs for model training
    private static int numOfHiddenNodes = 50; // number of hidden nodes for classification model
    private static INDArray weightsArray = Nd4j.create(new double[]{1.0, 0.43}); // weights array for weighted loss function for imbalance data.

    public static void main(String[] args) throws Exception{
        // 1. ======== load data ========
        RecordReader data = load_data("/germanCredit/germanCredit.csv"); // call the function to load data


        // 2. ======== transform data ========
        List<List<Writable>> transformedData = transform_data(data); // call the function to transform data


        // 3. ======== shuffle data ========
        DataSet dataset = shuffle_data(transformedData); // call the function to shuffle data


        // 4. ======== split train/test set with 70:30 ratio (train:test) and create iterators ========
        SplitTestAndTrain testAndTrain = dataset.splitTestAndTrain(0.7); // use 70% of the whole data for training set and 30% for test set

        DataSet trainSet = testAndTrain.getTrain();
        DataSet testSet = testAndTrain.getTest();

        // create data iterators for train set set and test set
        DataSetIterator trainIterator = new ViewIterator(trainSet, trainSet.numExamples());
        DataSetIterator testIterator = new ViewIterator(testSet, testSet.numExamples());


        // 5. ======== build and initialise the model ========
        MultiLayerNetwork model = build_model(weightsArray); // call the function to build the model
        model.init();


        // 6. ======== configure listener (UI & Training Loss Value) ========
        StatsStorage statsStorage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(statsStorage);
        model.setListeners(
                new StatsListener( statsStorage),
                new ScoreIterationListener(10)
        );

        // 7. ======== fit the model ========
        // **Solution 1**: early stopping - get optimal number of Epochs from performing early stopping
        // Add your code here















        // This is for training without Early Stopping
        // declare variables for model evaluation during training
        Evaluation evalTrain;
        Evaluation evalValid;
        DataSetLossCalculator trainLossCalculator = new DataSetLossCalculator(trainIterator, true);
        DataSetLossCalculator validLossCalculator = new DataSetLossCalculator(testIterator, true);
        ArrayList<Double> trainingLoss = new ArrayList<>();
        ArrayList<Double> validationLoss = new ArrayList<>();

        // fit model
        for(int i = 0; i<numOfEpochs; i++){
            model.fit(trainIterator); // fit the train set to the model

            // logging of model performance
            trainingLoss.add(trainLossCalculator.calculateScore(model)); // calculate training loss and add to trainingLoss ArrayList
            validationLoss.add(validLossCalculator.calculateScore(model)); // calculate validation loss and add to validationLoss ArrayList

            evalTrain = model.evaluate(trainIterator); // evaluate the model using train set
            evalValid = model.evaluate(testIterator); // evaluate the model using test set as the validation set -> if there is a bigger dataset, it is better to not use the test set as the validation set
            System.out.println("EPOCH: " + i + ", Train f1: " + evalTrain.f1());
            System.out.println("EPOCH: " + i + ", Validation f1: " + evalValid.f1());

            trainIterator.reset(); // reset train iterator back to the beginning
            testIterator.reset(); // reset test iterator back to the beginning
        }


        // 8. ======== evaluate model ========
        // plot training/validation loss graph
        plotLossGraph("Number of Epochs", "Training/Validation Loss", trainingLoss, validationLoss, numOfEpochs);

        // evaluate on train set
        Evaluation evalTrainSet = model.evaluate(trainIterator);
        System.out.print("Evaluation on Train Set");
        System.out.println(evalTrainSet.stats());

        // if there is a bigger dataset, it is better to evaluate on a separate test set that your model has never seen before during training
        Evaluation evalTestSet = model.evaluate(testIterator);
        System.out.print("Evaluation on Test Set");
        System.out.println(evalTestSet.stats());
    }




    /*
     * =================================
     *  List of functions to be used
     * =================================
     */

    /*
     ************** 1. function to load data **************
     */
    private static RecordReader load_data(String filepath) throws Exception{

        // set file path
        File dataFile = new ClassPathResource(filepath).getFile();

        // split file
        FileSplit fileSplit = new FileSplit(dataFile);

        // set CSV Record Reader and initialize it
        RecordReader rr = new CSVRecordReader(0,' ');
        rr.initialize(fileSplit);

        return rr;
    }


    /*
     ************** 2. function to transform data **************
     */
    private static List<List<Writable>> transform_data(RecordReader data){

        // build a schema to define the layout of the tabular data
        Schema schema = new Schema.Builder()
                .addColumnCategorical("Status_of_Existing_Checking_Account", "A11", "A12", "A13", "A14")
                .addColumnInteger("Duration_in_Month")
                .addColumnCategorical("Credit_History", "A30", "A31", "A32", "A33", "A34")
                .addColumnCategorical("Purpose", "A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49", "A410")
                .addColumnDouble("Credit_Amount")
                .addColumnCategorical("Savings_Account", "A61", "A62", "A63", "A64", "A65")
                .addColumnCategorical("Present_Employment_Since", "A71", "A72", "A73", "A74", "A75")
                .addColumnDouble("Installment_Rate")
                .addColumnCategorical("Personal_Status_and_Sex", "A91", "A92", "A93", "A94", "A95")
                .addColumnCategorical("Other_Debtors_or_Guarantors", "A101", "A102", "A103")
                .addColumnInteger("Present_Residence_Since")
                .addColumnCategorical("Property", "A121", "A122", "A123", "A124")
                .addColumnInteger("Age_in_Years")
                .addColumnCategorical("Other_Installment_Plans", "A141", "A142", "A143")
                .addColumnCategorical("Housing", "A151", "A152", "A153")
                .addColumnInteger("Number_of_Existing_Credits")
                .addColumnCategorical("Job", "A171", "A172", "A173", "A174")
                .addColumnInteger("Number_of_People_to_Provide_Maintenance")
                .addColumnCategorical("Telephone", "A191", "A192")
                .addColumnCategorical("Foreign_Worker", "A201", "A202")
                .addColumnCategorical("Customer_Credit", "1", "2")
                .build();

        // data transformation
        TransformProcess transformProcess = new TransformProcess.Builder(schema)
                .categoricalToOneHot("Status_of_Existing_Checking_Account","Credit_History","Purpose","Savings_Account","Present_Employment_Since", "Personal_Status_and_Sex",
                        "Other_Debtors_or_Guarantors", "Property", "Other_Installment_Plans", "Housing", "Job", "Telephone" ,"Foreign_Worker" )
                .conditionalReplaceValueTransform(
                        "Customer_Credit",     //Column to operate on
                        new IntWritable(0),    //New value to use, when the condition is satisfied
                        new IntegerColumnCondition("Customer_Credit", ConditionOp.Equal, 1)) //Customer_Credit: Good == 0
                .conditionalReplaceValueTransform(
                        "Customer_Credit",     //Column to operate on
                        new IntWritable(1),    //New value to use, when the condition is satisfied
                        new IntegerColumnCondition("Customer_Credit", ConditionOp.Equal, 2)) //Customer_Credit: Bad == 1, since bad is more important
                .build();

        // process the data
        List<List<Writable>> allData = new ArrayList<>();
        while(data.hasNext()){
            allData.add(data.next());
        }

        // apply TransformProcess to schema
        List<List<Writable>> processedData = LocalTransformExecutor.execute(allData, transformProcess);

        return processedData;
    }


    /*
     ************** 3. function to shuffle_data dataset using dataset iterators **************
     */
    private static DataSet shuffle_data(List<List<Writable>> transformedData){ //

        // use collection record reader to read the transformed data in List<List<Writable>> type
        CollectionRecordReader collectionRecordReader = new CollectionRecordReader(transformedData);

        // create dataset Iterator with global variables batch size, label index, and number of classes/labels
        DataSetIterator iterator = new RecordReaderDataSetIterator(collectionRecordReader, transformedData.size(), labelIndex, numOfClasses);

        // shuffle the data
        DataSet dataSet = iterator.next();
        dataSet.shuffle();

        // compute statistics and normalise all the variables
        DataNormalization normalizer = new NormalizerMinMaxScaler();
        normalizer.fit(dataSet); // Collect the statistics (mean/stddev) from the training data. This does not modify the input data
        normalizer.transform(dataSet); // Apply normalization to the dataset

        // print out the feature shape in [numOfRows, numOfCols]
        INDArray features = dataSet.getFeatures();
        System.out.println("\nFeature shape: " + features.shapeInfoToString() + "\n");

        return dataSet;
    }



    /*
     ************** 5. function to build model **************
     */
    private static MultiLayerNetwork build_model(INDArray weightsArray){

        // build a MLP model
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(1234)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.001, Nesterovs.DEFAULT_NESTEROV_MOMENTUM)) // Gradient Descent optimization algorithm
                // **SOLUTION 2**: add regularisation
                // Add your code here


                .list()
                .layer(new DenseLayer.Builder()
                .activation(Activation.RELU)
                .nIn(numOfFeatures)
                .nOut(numOfHiddenNodes)
                .build())
                .layer(new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nIn(numOfHiddenNodes)
                        .nOut(numOfHiddenNodes)
                        .build())
                // **SOLUTION 3**: Reduce the number of hidden layers for less complex tasks
                // comment out the hidden layers below to reduce complexity
                .layer(new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nIn(numOfHiddenNodes)
                        .nOut(numOfHiddenNodes)
                        .build())
                .layer(new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nIn(numOfHiddenNodes)
                        .nOut(numOfHiddenNodes)
                        .build())
                .layer(new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nIn(numOfHiddenNodes)
                        .nOut(numOfHiddenNodes)
                        .build())
                // **SOLUTION 4** dropout layers (removing nodes from the network)
                // Add your code here


                .layer(new OutputLayer.Builder()
                        .lossFunction(new LossBinaryXENT(weightsArray)) // loss function for binary classifier
                        .activation(Activation.SIGMOID)
                        .nIn(numOfHiddenNodes)
                        .nOut(2)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);


        return model;
    }


}
