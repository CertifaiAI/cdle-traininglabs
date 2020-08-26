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

package ai.certifai.solution.datavec.loadcsv;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.MathFunction;
import org.datavec.api.transform.MathOp;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.condition.column.InvalidValueColumnCondition;
import org.datavec.api.transform.filter.ConditionFilter;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class LoadCSVHousePrice {
    private static final int NUMLINESTOSKIP = 1;
    private static final int EPOCHS = 200;
    private static final int SEED = 100;
    private static final double LEARNINGRATE = 0.1;

    public static void main(String[] args) throws IOException, InterruptedException {
        //  Step 1 :    Input the file and load into record reader
        //  Always take a look at the data and the its description (if available) before starting
        //  In this case, the data and its description are stored in the resource/datavec/houseprice
        File inputFile = new ClassPathResource("datavec/houseprice/housePrice.csv").getFile();
        RecordReader CSVreader = new CSVRecordReader(NUMLINESTOSKIP);
        CSVreader.initialize(new FileSplit(inputFile));

        //  Step 2  :   Build up the schema of the data set by referring with the csv file
        Schema schema = new Schema.Builder()
                .addColumnString("Id")
                .addColumnCategorical("MSZoning", Arrays.asList("A", "C", "FV", "I", "RH", "RL", "RP", "RM"))
                .addColumnsInteger(
                        "LotFrontage",
                        "OverallQual",
                        "MasVnrArea",
                        "TotalBsmtSF",
                        "1stFlrSF",
                        "2ndFlrSF",
                        "GrLivArea",
                        "Fireplaces")
                .addColumnCategorical("GarageType", Arrays.asList("2Types", "Attchd", "Basment", "BuiltIn", "CarPort", "Detchd", "NA"))
                .addColumnsInteger("GarageCar", "GarageArea", "OpenPorchSF")
                .addColumnCategorical("PoolQC", Arrays.asList("Ex", "Gd", "TA", "Fa", "NA"))
                .addColumnCategorical("SaleCondition", Arrays.asList("Normal", "Abnorml", "AdjLand", "Alloca", "Family", "Partial"))
                .addColumnDouble("SalesPrice")
                .build();

        System.out.println("*****************************   Before Transform Process    *****************************");
        System.out.println(schema);

        //  Step 3  Using the transform process for preprocessing the dataset
        TransformProcess transformProcess = new TransformProcess.Builder(schema)
                //  Remove the Id column which has all unique values that are not needed in training
                .removeColumns("Id")
                //  Perform log transform for the sales price
                .doubleMathOp("SalesPrice", MathOp.Add, 1)
                .doubleMathFunction("SalesPrice", MathFunction.LOG)
                //  The data in csv file are different from the data_description file, standardize the value to prevent confusion
                .stringMapTransform("MSZoning", Collections.singletonMap("C (all)", "C"))
                .stringToCategorical("MSZoning", Arrays.asList("A", "C", "FV", "I", "RH", "RL", "RP", "RM"))
                //  Removing the NA values in the dataset (from columns "LotFrontage" and "MasVnrArea") to prevent error when training
                .filter(new ConditionFilter(new InvalidValueColumnCondition("LotFrontage")))
                .filter(new ConditionFilter(new InvalidValueColumnCondition("MasVnrArea")))
                //  In this case, we don't remove the NA values in the GarageType and PoolQC as according to the description, NA represent None
                //  One Hot Encode the categorical features so that the machine could understand categorical features
                .categoricalToOneHot("MSZoning")
                .categoricalToOneHot("GarageType")
                .categoricalToOneHot("PoolQC")
                .categoricalToOneHot("SaleCondition")
                .build();

        Schema finalSchema = transformProcess.getFinalSchema();
        System.out.println("*****************************   After TransformProcess   *****************************");
        System.out.println(finalSchema);

        //  Step 4 Transform the schema
        //  Method 1 :  Using LocalTransformExecutor
        List<List<Writable>> originalData = new ArrayList<>();
        while (CSVreader.hasNext()) {
            originalData.add(CSVreader.next());
        }

        List<List<Writable>> processedData = LocalTransformExecutor.execute(originalData, transformProcess);
        RecordReader collectionRecordReader = new CollectionRecordReader(processedData);
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(collectionRecordReader, originalData.size(), 37, 37, true);

        //  Method 2 :  Using TransformProcessRecordReader
//        TransformProcessRecordReader transformReader = new TransformProcessRecordReader(CSVreader,transformProcess);
//        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(transformReader,2000,37,37,true);

//        //  OPTIONAL - Uncomment the method below to start training
        modelTraining(dataSetIterator);
    }

    public static void modelTraining(DataSetIterator dataSetIterator) {
        DataSet allData = dataSetIterator.next();
        allData.shuffle(SEED);

        SplitTestAndTrain testTrainSplit = allData.splitTestAndTrain(0.7);

        DataSet trainingSet = testTrainSplit.getTrain();
        DataSet testSet = testTrainSplit.getTest();

        ViewIterator trainIter = new ViewIterator(trainingSet, trainingSet.numExamples());
        ViewIterator testIter = new ViewIterator(testSet, testSet.numExamples());

        //  Scale the data set to optimize the training process
        DataNormalization normalizer = new NormalizerMinMaxScaler();
        normalizer.fit(trainIter);
        trainIter.setPreProcessor(normalizer);
        testIter.setPreProcessor(normalizer);

        //  Configuring the structure of the NN
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(SEED)
                .updater(new Adam(LEARNINGRATE))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new OutputLayer.Builder()
                        .nIn(trainingSet.numInputs())
                        .nOut(1)
                        .activation(Activation.IDENTITY)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(50));

        //  Fitting the model for nEpochs
        model.fit(trainIter, EPOCHS);

        INDArray predict = model.output(testIter);
        System.out.println("Predicted" + "\t" + "Ground Truth");
        for (int i = 0; i < predict.length(); i++) {
            System.out.println(predict.getRow(i) + "\t" + testSet.getLabels().getRow(i));
        }

        //  Evaluating the outcome of our trained model
        RegressionEvaluation regEval = model.evaluateRegression(testIter);
        System.out.println(regEval.stats());
    }
}
