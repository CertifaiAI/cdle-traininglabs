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

package ai.certifai.solution.segmentation.car;

import ai.certifai.utilities.Visualization;
import ai.certifai.Helper;
import org.datavec.image.transform.*;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.CnnLossLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.UNet;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;
import org.slf4j.Logger;

import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_imgproc.CV_RGB2GRAY;

public class PretrainedUNET {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(
            PretrainedUNET.class);

    private static final String featurizeExtractionLayer = "conv2d_4";
    private static final long seed = 12345;
    private static final int nEpochs = 5;
    private static final int height = 224;
    private static final int width = 224;
    private static final int batchSize = 2;
    private static final double trainPerc = 0.8;
    private static String modelExportDir;

    public static void main(String[] args) throws IOException {

        /*
         * Instructions for this lab exercise:
         * STEP 1: Import pretrained UNET (provided in model zoo).
         * STEP 2: Configuration of transfer learning.
         * STEP 3: Load data into RecordReaderDataSetIterator.
         * STEP 4: Run training.
         * STEP 5: We will use IOU (Intersection Over Union) as our evaluation metric. Complete the code for IOU calculation.
         * STEP 6: Hyperparameter Tuning. (Epochs, learning rate etc.)
         *
         * */

        //STEP 1: Import pretrained UNET (provided in model zoo)
        ZooModel zooModel = UNet.builder().build();

        ComputationGraph unet = (ComputationGraph) zooModel.initPretrained(PretrainedType.SEGMENT);
        System.out.println(unet.summary());

        // Set listeners
        StatsStorage statsStorage = new InMemoryStatsStorage();
        StatsListener statsListener = new StatsListener(statsStorage);
        ScoreIterationListener scoreIterationListener = new ScoreIterationListener(1);

        //STEP 2: Configuration of transfer learning
        //STEP 2.1: Set updater and learning rate)
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .updater(new Adam(new StepSchedule(ScheduleType.EPOCH, 3e-4, 0.5, 5)))
                .seed(seed)
                .build();

        //Construct a new model with the intended architecture and print summary
        //STEP 2.2: Set which pre-trained layer to freeze and use as feature extractor
        //STEP 2.3: Add a CnnLossLayer to form a Fully Convolutional Network
        ComputationGraph unetTransfer = new TransferLearning.GraphBuilder(unet)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor(featurizeExtractionLayer)
                .removeVertexAndConnections("activation_23")
                .nInReplace("conv2d_1", 1, WeightInit.XAVIER)
                .nOutReplace("conv2d_23", 1, WeightInit.XAVIER)
                .addLayer("output",
                        new CnnLossLayer.Builder(LossFunctions.LossFunction.XENT)
                                .activation(Activation.SIGMOID).build(),
                        "conv2d_23")
                .setOutputs("output")
                .build();

        System.out.println(unetTransfer.summary());

        unetTransfer.setListeners(statsListener, scoreIterationListener);

//        Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();
        uiServer.attach(statsStorage);

        // STEP 3: Load data into RecordReaderDataSetIterator
        CarDataSetIterator.setup(batchSize, trainPerc, getImageTransform());

        //Create iterators
        RecordReaderDataSetIterator imageDataSetTrain = CarDataSetIterator.trainIterator();
        RecordReaderDataSetIterator imageDataSetVal = CarDataSetIterator.valIterator();

        // Visualisation -  training
        JFrame frame = Visualization.initFrame("Viz");
        JPanel panel = Visualization.initPanel(
                frame,
                batchSize,
                height,
                width,
                1
        );

        //STEP 4: Run training
        for (int i = 0; i < nEpochs; i++) {

            log.info("Epoch: " + i);

            while (imageDataSetTrain.hasNext()) {
                DataSet imageSet = imageDataSetTrain.next();

                unetTransfer.fit(imageSet);

                INDArray predict = unetTransfer.output(imageSet.getFeatures())[0];

                Visualization.visualize(
                        imageSet.getFeatures(),
                        imageSet.getLabels(),
                        predict,
                        frame,
                        panel,
                        batchSize,
                        224,
                        224
                );
            }

            imageDataSetTrain.reset();
        }


        // VALIDATION
        Evaluation eval = new Evaluation(2);

        // EXPORT IMAGES
         File exportDir = Paths.get(
                System.getProperty("user.home"),
                Helper.getPropValues("dl4j_home.export-images")
        ).toFile();

        if (!exportDir.exists()) {
            exportDir.mkdir();
        }

        float IOUTotal = 0;
        int count = 0;
        while (imageDataSetVal.hasNext()) {
            DataSet imageSetVal = imageDataSetVal.next();

            INDArray predictVal = unetTransfer.output(imageSetVal.getFeatures())[0];
            INDArray labels = imageSetVal.getLabels();

//            // Uncomment the following if there's a need to export images
//            if (count % 5 == 0) {
//                Visualization.export(exportDir, imageSetVal.getFeatures(), imageSetVal.getLabels(), predictVal, count);
//            }

            count++;

            eval.eval(labels, predictVal);


            //STEP 5: Complete the code for IOU calculation here
            //Intersection over Union:  TP / (TP + FN + FP)
            float IOUCar = (float) eval.truePositives().get(1) / ((float) eval.truePositives().get(1) + (float) eval.falsePositives().get(1) + (float) eval.falseNegatives().get(1));
            IOUTotal = IOUTotal + IOUCar;

            System.out.println("IOU Car " + String.format("%.3f", IOUCar));

            eval.reset();

            Visualization.visualize(
                    imageSetVal.getFeatures(),
                    imageSetVal.getLabels(),
                    predictVal,
                    frame,
                    panel,
                    batchSize,
                    height,
                    width
            );

        }

        System.out.println("Mean IOU: " + IOUTotal / count);

        // WRITE MODEL TO DISK
        modelExportDir = Paths.get(
                System.getProperty("user.home"),
                Helper.getPropValues("dl4j_home.generated-models")
        ).toString();


        File locationToSaveModel = new File(Paths.get(modelExportDir, "segmentUNET.zip").toString());
        if (!locationToSaveModel.exists()) {
            locationToSaveModel.getParentFile().mkdirs();
        }


        ModelSerializer.writeModel(unetTransfer, locationToSaveModel, false);
        log.info("Model saved");

    }


    public static ImageTransform getImageTransform() {
        ImageTransform rgb2gray = new ColorConversionTransform(CV_RGB2GRAY);

        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(
                new Pair<>(rgb2gray, 1.0)
        );
        return new PipelineImageTransform(pipeline, false);
    }
}