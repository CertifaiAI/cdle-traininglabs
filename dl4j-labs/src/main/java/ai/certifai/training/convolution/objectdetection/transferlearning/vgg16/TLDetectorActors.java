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

package ai.certifai.training.convolution.objectdetection.transferlearning.vgg16;

import ai.certifai.training.convolution.objectdetection.transferlearning.vgg16.dataHelpers.ActorsDatasetIterator;
import com.google.common.primitives.Ints;
import org.datavec.api.records.metadata.RecordMetaDataURI;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.Arrays;

/*
 * Another transfer learning example with VGG16 (VGGFace)
 */
public class TLDetectorActors {
    private static final Logger log = LoggerFactory.getLogger(TLDetectorActors.class);
    private static ComputationGraph model;
    private static String modelFilename = new File(".").getAbsolutePath() + "/generated-models/VGG16_TLDetectorActors.zip";

    // parameters for the training phase
    private static int batchSize = 5;
    private static int nEpochs = 40;
    private static double learningRate = 1e-3;
    private static int nClasses = 3;
    private static List<String> labels;
    private static int seed = 123;

    public static void main(String[] args) throws Exception {

        // Directory for Custom train and test datasets
        log.info("Load data...");
        ActorsDatasetIterator.setup(new ClassPathResource("actors/vgg16_244").getFile(), batchSize,70);
        RecordReaderDataSetIterator trainIter = ActorsDatasetIterator.trainIterator();
        trainIter.setPreProcessor( new VGG16ImagePreProcessor());

        RecordReaderDataSetIterator testIter = ActorsDatasetIterator.testIterator();
        testIter.setPreProcessor( new VGG16ImagePreProcessor());

        // Print Labels
        labels = trainIter.getLabels();
        System.out.println(Arrays.toString(labels.toArray()));

        if (new File(modelFilename).exists()) {

            // Load trained model from previous execution
            log.info("Load model...");
            model = ModelSerializer.restoreComputationGraph(modelFilename);

        } else {

            log.info("Build model...");
            // Load pretrained VGG16 model
            ComputationGraph pretrained = (ComputationGraph) VGG16.builder().build().initPretrained(PretrainedType.VGGFACE);
            log.info(pretrained.summary());

            // Transfer Learning steps - Model Configurations.
            FineTuneConfiguration fineTuneConf = getFineTuneConfiguration();

            // Transfer Learning steps - Modify prebuilt model's architecture for current scenario
            model = buildComputationGraph(pretrained, fineTuneConf);

            log.info("Train model...");

            UIServer server = UIServer.getInstance();
            StatsStorage storage = new InMemoryStatsStorage();
            server.attach(storage);
            model.setListeners(new ScoreIterationListener(1), new StatsListener(storage));

            for (int i = 0; i < nEpochs; i++) {
                trainIter.reset();
                while (trainIter.hasNext()) {
                    model.fit(trainIter.next());
                }
                log.info("*** Completed epoch {} ***", i);
            }
            ModelSerializer.writeModel(model, modelFilename, true);
        }

        validationTestDataset(testIter);
    }

    private static void validationTestDataset(RecordReaderDataSetIterator test) throws InterruptedException, IOException {

        test.setCollectMetaData(true);
        while (test.hasNext()) {
            DataSet ds = test.next();
            RecordMetaDataURI metadata = (RecordMetaDataURI) ds.getExampleMetaData().get(0);
            INDArray image = ds.getFeatures();
            System.out.println("label: " + labels.get(Ints.asList(ds.getLabels().toIntVector()).indexOf(1)));
            System.out.println(metadata.getURI());
            getPredictions(image);
        }
    }

    private static void getPredictions(INDArray image) throws IOException {
        INDArray[] output = model.output(false, image);
        List<Prediction> predictions = decodePredictions(output[0], 3);
        System.out.println("prediction: ");
        System.out.println(predictionsToString(predictions));
    }

    private static String predictionsToString(List<Prediction> predictions) {
        StringBuilder builder = new StringBuilder();
        for (Prediction prediction : predictions) {
            builder.append(prediction.toString());
            builder.append('\n');
        }
        return builder.toString();
    }

    private static List<Prediction> decodePredictions(INDArray encodedPredictions, int numPredicted) throws IOException {
        List<Prediction> decodedPredictions = new ArrayList<>();
        int[] topX = new int[numPredicted];
        float[] topXProb = new float[numPredicted];

        int i = 0;
        for (INDArray currentBatch = encodedPredictions.getRow(0).dup(); i < numPredicted; ++i) {

            topX[i] = Nd4j.argMax(currentBatch, 1).getInt(0);
            topXProb[i] = currentBatch.getFloat(0, topX[i]);
            currentBatch.putScalar(0, topX[i], 0.0D);
            decodedPredictions.add(new Prediction(labels.get(topX[i]), (topXProb[i] * 100.0F)));
        }
        return decodedPredictions;
    }

    public static class Prediction {

        private String label;
        private double percentage;

        public Prediction(String label, double percentage) {
            this.label = label;
            this.percentage = percentage;
        }

        public void setLabel(final String label) {
            this.label = label;
        }

        public String toString() {
            return String.format("%s: %.2f ", this.label, this.percentage);
        }
    }

    private static ComputationGraph buildComputationGraph(ComputationGraph pretrained, FineTuneConfiguration fineTuneConf) {
        //Construct a new model with the intended architecture and print summary
        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(pretrained)
                .fineTuneConfiguration(fineTuneConf)
                .removeVertexKeepConnections("predictions")
                .addLayer("predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(2622).nOut(nClasses)
                                .weightInit(new NormalDistribution(0,0.2*(2.0/(4096+nClasses)))) //This weight init dist gave better results than Xavier
                                .activation(Activation.SOFTMAX).build(),
                        "fc8")
                .setOutputs("predictions")
                .build();
        log.info(vgg16Transfer.summary());

        return vgg16Transfer;
    }

    private static FineTuneConfiguration getFineTuneConfiguration() {

        FineTuneConfiguration _FineTuneConfiguration = new FineTuneConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(1.0)
                .updater(new Nesterovs.Builder().learningRate(learningRate).momentum(Nesterovs.DEFAULT_NESTEROV_MOMENTUM).build())
                .l2(0.00001)
                .activation(Activation.IDENTITY)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .build();

        return _FineTuneConfiguration;
    }
}
