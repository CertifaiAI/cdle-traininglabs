/*
 * Copyright (c) 2019 Skymind Holdings Bhd.
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

package ai.certifai.training.dl1_datavec.loadcsv;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.common.io.ClassPathResource;

import java.io.File;
import java.util.Arrays;

public class LoadCSV1 {
    private static int numLinesToSkip = 0;
    private static char delimiter = ',';

    private static int batchSize = 150; // Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)
    private static int labelIndex = 4; // index of label/class column
    private static int numClasses = 3; // number of class in iris dataset

    public static void main(String[] args) throws Exception {
        // define csv file location
        File inputFile = new ClassPathResource("datavec/iris.txt").getFile();
        FileSplit fileSplit = new FileSplit(inputFile);

        // get dataset using record reader. CSVRecordReader handles loading/parsing
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReader.initialize(fileSplit);

        // create iterator from record reader
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);
        DataSet allData = iterator.next();

        System.out.println("Shape of allData vector:");
        System.out.println(Arrays.toString(allData.getFeatures().shape()));

        // shuffle and split all data into training and test set
        allData.shuffle();
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.8);
        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        System.out.println("\nShape of training vector:");
        System.out.println(Arrays.toString(trainingData.getFeatures().shape()));
        System.out.println("\nShape of test vector:");
        System.out.println(Arrays.toString(testData.getFeatures().shape()));

        // create iterator for splitted training and test dataset
        DataSetIterator trainIterator = new ViewIterator(trainingData, 4);
        DataSetIterator testIterator = new ViewIterator(testData, 2);

        // normalize data to 0 - 1
        DataNormalization scaler = new NormalizerMinMaxScaler(0,1);
        scaler.fit(trainIterator);
        trainIterator.setPreProcessor(scaler);
        testIterator.setPreProcessor(scaler);

        System.out.println("\nShape of training batch vector:");
        System.out.println(Arrays.toString(trainIterator.next().getFeatures().shape()));
        System.out.println("\nShape of test batch vector:");
        System.out.println(Arrays.toString(testIterator.next().getFeatures().shape()));
        System.out.println("\ntraining vector: ");
        System.out.println(trainIterator.next().getFeatures());
        System.out.println("\ntest vector: ");
        System.out.println(testIterator.next().getFeatures());
    }
}
