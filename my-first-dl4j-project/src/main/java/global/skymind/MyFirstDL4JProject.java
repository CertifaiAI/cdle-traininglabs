/*******************************************************************************
 * Copyright (c) 2020 CertifAI Sdn. Bhd.
 *
 *  This program and the accompanying materials are made available under the
 *  terms of the Apache License, Version 2.0 which is available at
 *  https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  License for the specific language governing permissions and limitations
 *  under the License.
 *
 *  SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package global.skymind;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;


/**
 * Simple kickstart project to train simple MultiLayerNetwork.
 *
 * @author ChiaWei Lim
 */
public class MyFirstDL4JProject
{
    private static int numLinesToSkip = 0;
    private static char delimiter = ',';

    private static int batchSize = 64; // Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)
    private static int labelIndex = 4; // index of label/class column
    private static int numClasses = 3; // number of class in iris dataset

    private static final Logger log = LoggerFactory.getLogger(MyFirstDL4JProject.class);

    public static void main( String[] args ) throws Exception {
        // define csv file location
        File inputFile = new ClassPathResource("iris.txt").getFile();
        FileSplit fileSplit = new FileSplit(inputFile);

        // get dataset using record reader. CSVRecordReader handles loading/parsing
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReader.initialize(fileSplit);

        // create iterator from record reader
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(Nesterovs.DEFAULT_NESTEROV_LEARNING_RATE))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(4)
                        .nOut(100)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(config);

        network.init();
        network.setListeners(new ScoreIterationListener(1));

        network.fit(iterator, 10);

        log.info("Program end...");
    }
}
