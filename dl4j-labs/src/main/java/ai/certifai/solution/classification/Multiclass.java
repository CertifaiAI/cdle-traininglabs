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

package ai.certifai.solution.classification;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * Multiclass classification example
 *
 * @author devenyantis
 */
public class Multiclass {

    private static final int totalData = 42151;
    private static final double ratioTrainTestSplit = 0.8;

    // Training info
    private static final int epoch = 1000;

    public static void main(String[] args) throws IOException, InterruptedException {

        //=====================================================================
        //            Step 1: Load & Transform data
        //=====================================================================

        RecordReader rr = loadData();

        List<List<Writable>> rawTrainData = new ArrayList<>();
        List<List<Writable>> rawTestData = new ArrayList<>();

        // Get total length of data
        int numTrainData = (int) Math.round(ratioTrainTestSplit * totalData);
        int idx = 0;
        while (rr.hasNext()) {
            if(idx < numTrainData) {
                rawTrainData.add(rr.next());
            } else {
                rawTestData.add(rr.next());
            }
            idx++;
        }

        System.out.println("Total train Data " + rawTrainData.size());
        System.out.println("Total test Data " + rawTestData.size());

        List<List<Writable>> transformedTrainData = transformData(rawTrainData);
        List<List<Writable>> transformedTestData = transformData(rawTestData);

        DataSetIterator trainData = makeIterator(transformedTrainData);
        DataSetIterator testData = makeIterator(transformedTestData);

        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainData);
        trainData.setPreProcessor(normalizer);
        testData.setPreProcessor(normalizer);

        //=====================================================================
        //            Step 2: Define Model
        //=====================================================================

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(1234)
                .updater(new Nesterovs(0.001, Nesterovs.DEFAULT_NESTEROV_MOMENTUM))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nIn(6)
                        .nOut(32)
                        .build())
                .layer(1, new DropoutLayer(0.3))
                .layer(2, new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nOut(64)
                        .build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nOut(7)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        //=====================================================================
        //            Step 3: Set Listener
        //=====================================================================

        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);

        // Set model listeners
        model.setListeners(new StatsListener(storage, 10));

        //=====================================================================
        //            Step 4: Train model
        //=====================================================================

        Evaluation eval;
        for(int i=0; i < epoch; i++) {
            model.fit(trainData);
            eval = model.evaluate(testData);
            System.out.println("EPOCH: " + i + " Accuracy: " + eval.accuracy());
            testData.reset();
            trainData.reset();
        }

        System.out.println("=== Train data evaluation ===");
        eval = model.evaluate(trainData);
        System.out.println(eval.stats());

        System.out.println("=== Test data evaluation ===");
        eval = model.evaluate(testData);
        System.out.println(eval.stats());

    }

    private static RecordReader loadData() throws IOException, InterruptedException {

        int numLinesToSkip = 1; // how many rows to skip. Skip header row.
        char delimiter = ',';

        // Define csv location
        File inputFile = new ClassPathResource("TabularData/AReM.csv").getFile();
        FileSplit fileSplit = new FileSplit(inputFile);

        // Read dataset using record reader. CSVRecordReader handles loading/parsing
        RecordReader rr = new CSVRecordReader(numLinesToSkip, delimiter);
        rr.initialize(fileSplit);

        return rr;
    }

    private static List<List<Writable>> transformData(List<List<Writable>> data) {

        //=====================================================================
        //            Define Input data schema
        //=====================================================================

        Schema inputDataSchema = new Schema.Builder()
                .addColumnsFloat("avg_rss12", "var_rss12", "avg_rss13", "var_rss13", "avg_rss23", "var_rss23")
                .addColumnCategorical("class", Arrays.asList("walking","standing", "cycling", "sitting", "lying", "bending1", "bending2"))
                .build();

        // print data Schema
        System.out.println(inputDataSchema);

        //=====================================================================
        //            Define transformation operations
        //=====================================================================

        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                .categoricalToInteger("class")
                .build();

        //=====================================================================
        //            Perform transformation
        //=====================================================================

        return LocalTransformExecutor.execute(data, tp);
    }

    private static DataSetIterator makeIterator(List<List<Writable>> data) {

        // Data info
        int labelIndex = 6; // Index of column of the labels
        int numClasses = 7; // Number of unique classes for the labels

        RecordReader collectionRecordReaderData = new CollectionRecordReader(data);

        return new RecordReaderDataSetIterator(collectionRecordReaderData, data.size(), labelIndex, numClasses);
    }
}
