/*
 *
 *  * ******************************************************************************
 *  *  * Copyright (c) 2019 Skymind AI Bhd.
 *  *  * Copyright (c) 2020 CertifAI Sdn. Bhd.
 *  *  *
 *  *  * This program and the accompanying materials are made available under the
 *  *  * terms of the Apache License, Version 2.0 which is available at
 *  *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *  *
 *  *  * Unless required by applicable law or agreed to in writing, software
 *  *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  *  * License for the specific language governing permissions and limitations
 *  *  * under the License.
 *  *  *
 *  *  * SPDX-License-Identifier: Apache-2.0
 *  *  *****************************************************************************
 *
 *
 */

package ai.certifai.training.recurrent.physionet;

import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.nd4j.linalg.io.ClassPathResource;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.ROC;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.io.UnsupportedEncodingException;

/**

 Train LSTM for mortality classification of 0 and 1
 Data are in physionet2012/resampled for where the data
 are resampled to one hour intervals with values averaged over the one hour period.
 Each file will have 49 lines, one header line and one line with values for each of the
 time steps. This dataset only has two days worth of data per patient

 A flag is added, if data is not measured the flag is set to 1
 and the previous value is repeated.
 When a measured value is present the flag is set to 0.

 There are total of 4000 data.
 First 3200 samples for training.
 Consecutive 400 samples each for validation and testing.

 Data publicly available at https://physionet.org/challenge/2012/

 For more details on RNNs in DL4J, see the following:
 http://deeplearning4j.org/usingrnns
 http://deeplearning4j.org/lstm
 http://deeplearning4j.org/recurrentnetwork

 Look for LAB STEP below. Uncomment to proceed.
 1. Load the validation and testing data
 2. Save the model
 3. Evaluate the results
 4. Tune the network

 **/

public class PhysionetMultivariateTimeSeriesClassification
{

    public static final int trainSamples = 3200;
    public static final int validSamples = 400;
    public static final int testSamples = 400;

    public static final int miniBatchSize = 200;
    public static final int numLabelClasses = 2;

    public static void main(String[] args) throws IOException, InterruptedException
    {


        File baseDir = new ClassPathResource("physionet2012").getFile();
        File featuresDir = new File(baseDir, "resampled");
        File labelsDir= new File(baseDir, "mortality");

        //load training data
        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader(1, ",");
        trainFeatures.initialize(new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/%d.csv", 0, trainSamples - 1));

        SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
        trainLabels.initialize(new NumberedFileInputSplit(labelsDir.getAbsolutePath() + "/%d.csv", 0, trainSamples - 1));

        DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, numLabelClasses, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        /*
		#### LAB STEP 1 #####
		Load the testing data
		400 samples for validation and testing separately.
        */
        //load validation data
        SequenceRecordReader validFeatures = new CSVSequenceRecordReader(1, ",");
        validFeatures.initialize(new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/%d.csv", trainSamples, trainSamples + validSamples - 1));

        SequenceRecordReader validLabels = new CSVSequenceRecordReader();
        validLabels.initialize(new NumberedFileInputSplit(labelsDir.getAbsolutePath() + "/%d.csv", trainSamples, trainSamples + validSamples - 1));
        DataSetIterator validData = new SequenceRecordReaderDataSetIterator(validFeatures, validLabels, miniBatchSize, numLabelClasses, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);



        //load testing data

        int numInputs = trainData.inputColumns();
        int numClasses = 2; // 0 or 1 for mortality
        int epochs = 25;
        int seedNumber = 123;
        double learningRate = 0.01;

        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seedNumber)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(learningRate))
                .graphBuilder()
                .addInputs("trainFeatures")
                .setOutputs("predictMortality")
                .addLayer("layer0", new LSTM.Builder()
                                .nIn(numInputs)
                                .nOut(100)
                                .activation(Activation.TANH)
                                .build(),
                        "trainFeatures")
                .addLayer("predictMortality", new RnnOutputLayer.Builder()
                                .nIn(100)
                                .nOut(numClasses)
                                .lossFunction(LossFunctions.LossFunction.XENT)
                                .activation(Activation.SIGMOID)
                                .build(),
                        "layer0")
                .backpropType(BackpropType.Standard)
                .build();


        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);

        ComputationGraph model = new ComputationGraph(config);
        model.init();
        model.setListeners(new StatsListener(storage, 10));

        int evalStep = 5;

        for(int i = 0; i < epochs; ++i)
        {
            model.fit(trainData);

            if(i % evalStep == 0)
            {

                ROC roc = new ROC(100);
                while (validData.hasNext())
                {
                    DataSet batch = validData.next();
                    INDArray[] output = model.output(batch.getFeatures());
                    roc.evalTimeSeries(batch.getLabels(), output[0]);

                }

                System.out.println("EPOCH " + i + " VALID AUC: " + roc.calculateAUC());
                validData.reset();

                /*
                #### LAB STEP 2 #####
                Save the model
                */

            }

        }

        //ROC
        /*
        #### LAB STEP 3 #####
        Evaluate the results
        */

    }
}
