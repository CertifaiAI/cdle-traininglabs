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

package ai.certifai.solution.object_detection.MetalDefectsDetector;

import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.YOLO2;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.CV_8U;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.helper.opencv_core.RGB;

/**
 * Description:
 * This is an example of a metal surface defect detection using YOLOv2 architecture.
 * This example illustrate a model training process with transfer learning approach by fine tuning the last few layers of a YOLOv2 pretrained model
 * Data Source: http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html
 * NOTE: DUE TO THE MEMORY CONSTRAINT, THE MODEL NEEDS TO BE TRAINED USING CPU
 *
 * Objective:
 * To learn about how to remove different layers in pretrained model and replace them with the new ones.
 */

public class MetalSurfaceDefectDetector_YOLOv2 {
    private static final Logger log = LoggerFactory.getLogger(MetalSurfaceDefectDetector_YOLOv2.class);
    private static int seed = 123;
    private static double detectionThreshold = 0.1;
    private static int nBoxes = 5;
    private static double lambdaNoObj = 0.7;
    private static double lambdaCoord = 0.7;
    private static double[][] priorBoxes = {{1, 4}, {2.5, 6}, {3, 1}, {3.5, 8}, {4, 9}};

    private static int batchSize = 8;
    private static int nEpochs = 8;
    private static double learningRate = 1e-4;
    private static int nClasses = 6;
    private static List<String> labels;

    private static File modelFilename = new File(System.getProperty("user.dir"), "generated-models/MetalSurfaceDefects_yolov2_CompleteRetrain.zip");
    private static ComputationGraph model;
    public static final Scalar BLUE = RGB(0, 0, 255);
    public static final Scalar GREEN = RGB(0, 255, 0);
    public static final Scalar RED = RGB(255, 0, 0);
    public static final Scalar YELLOW = RGB(255, 225, 0);
    public static final Scalar PINK = RGB(255, 0, 225);
    public static final Scalar CYAN = RGB(0, 225, 225);
    public static Scalar[] colormap = {BLUE, GREEN, RED, YELLOW, PINK, CYAN};
    private static String labeltext = null;

    public static void main(String[] args) throws Exception {

        MetalDefectDataSetIterator.setup();

        //        STEP 1 : Create iterators
        RecordReaderDataSetIterator trainIter = MetalDefectDataSetIterator.trainIterator(batchSize);
        RecordReaderDataSetIterator testIter = MetalDefectDataSetIterator.testIterator(1);

        labels = trainIter.getLabels();

        //        If model does not exist, train the model, else directly go to model evaluation and then run real time object detection inference.
        if (modelFilename.exists()) {
            //        STEP 2 : Load trained model from previous execution
            Nd4j.getRandom().setSeed(seed);
            log.info("Load model...");
            model = ModelSerializer.restoreComputationGraph(modelFilename);
        } else {
            Nd4j.getRandom().setSeed(seed);
            ComputationGraph pretrained = null;
            FineTuneConfiguration fineTuneConf = null;
            INDArray priors = Nd4j.create(priorBoxes);
            //     STEP 2 : Train the model using Transfer Learning
            //     STEP 2.1: Transfer Learning steps - Load YOLO prebuilt model.
            log.info("Build model...");
            pretrained = (ComputationGraph) YOLO2.builder().build().initPretrained();

            //     STEP 2.2: Transfer Learning steps - Model Configurations.
            fineTuneConf = getFineTuneConfiguration();

            //     STEP 2.3: Transfer Learning steps - Modify prebuilt model's architecture
            model = getComputationGraph(pretrained, priors, fineTuneConf);
            System.out.println(model.summary(InputType.convolutional(
                    MetalDefectDataSetIterator.yoloheight,
                    MetalDefectDataSetIterator.yolowidth,
                    nClasses)));

            //     STEP 2.4: Training and Save model.
            log.info("Train model...");
            UIServer server = UIServer.getInstance();
            StatsStorage storage = new InMemoryStatsStorage();
            server.attach(storage);
            model.setListeners(new ScoreIterationListener(1), new StatsListener(storage));

            for (int i = 1; i < nEpochs + 1; i++) {
                trainIter.reset();
                while (trainIter.hasNext()) {
                    model.fit(trainIter.next());
                }
                log.info("*** Completed epoch {} ***", i);
            }
            ModelSerializer.writeModel(model, modelFilename, true);
            System.out.println("Model saved.");
        }
        //     STEP 3: Evaluate the model's accuracy by using the test iterator.
        OfflineValidationWithTestDataset(testIter);
    }

    private static ComputationGraph getComputationGraph(ComputationGraph pretrained, INDArray priors, FineTuneConfiguration fineTuneConf) {

        return new TransferLearning.GraphBuilder(pretrained)
                .fineTuneConfiguration(fineTuneConf)
                .removeVertexKeepConnections("conv2d_23")
                .removeVertexKeepConnections("outputs")
                .addLayer("conv2d_23",
                        new ConvolutionLayer.Builder(1, 1)
                                .nIn(1024)
                                .nOut(nBoxes * (5 + nClasses))
                                .stride(1, 1)
                                .convolutionMode(ConvolutionMode.Same)
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.IDENTITY)
                                .build(),
                        "leaky_re_lu_22")
                .addLayer("outputs",
                        new Yolo2OutputLayer.Builder()
                                .lambdaNoObj(lambdaNoObj)
                                .lambdaCoord(lambdaCoord)
                                .boundingBoxPriors(priors.castTo(DataType.FLOAT))
                                .build(),
                        "conv2d_23")
                .setOutputs("outputs")
                .build();
    }

    private static FineTuneConfiguration getFineTuneConfiguration() {

        return new FineTuneConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(1.0)
                .updater(new Adam.Builder().learningRate(learningRate).build())
                .l2(0.00001)
                .activation(Activation.IDENTITY)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .build();
    }

    //    Evaluate visually the performance of the trained object detection model
    private static void OfflineValidationWithTestDataset(RecordReaderDataSetIterator test) throws InterruptedException {
        NativeImageLoader imageLoader = new NativeImageLoader();
        CanvasFrame canvas = new CanvasFrame("Validate Test Dataset");
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) model.getOutputLayer(0);
        Mat convertedMat = new Mat();
        Mat convertedMat_big = new Mat();

        while (test.hasNext() && canvas.isVisible()) {

            org.nd4j.linalg.dataset.DataSet ds = test.next();
            INDArray features = ds.getFeatures();
            INDArray results = model.outputSingle(features);
            List<DetectedObject> objs = yout.getPredictedObjects(results, detectionThreshold);
            YoloUtils.nms(objs, 0.4);
            Mat mat = imageLoader.asMat(features);
            mat.convertTo(convertedMat, CV_8U, 255, 0);
            int w = mat.cols() * 2;
            int h = mat.rows() * 2;
            resize(convertedMat, convertedMat_big, new Size(w, h));
            convertedMat_big = drawResults(objs, convertedMat_big, w, h);
            canvas.showImage(converter.convert(convertedMat_big));
            canvas.waitKey();
        }
        canvas.dispose();
    }

    private static Mat drawResults(List<DetectedObject> objects, Mat mat, int w, int h) {
        for (DetectedObject obj : objects) {
            double[] xy1 = obj.getTopLeftXY();
            double[] xy2 = obj.getBottomRightXY();
            String label = labels.get(obj.getPredictedClass());
            int x1 = (int) Math.round(w * xy1[0] / MetalDefectDataSetIterator.gridWidth);
            int y1 = (int) Math.round(h * xy1[1] / MetalDefectDataSetIterator.gridHeight);
            int x2 = (int) Math.round(w * xy2[0] / MetalDefectDataSetIterator.gridWidth);
            int y2 = (int) Math.round(h * xy2[1] / MetalDefectDataSetIterator.gridHeight);
            //Draw bounding box
            rectangle(mat, new Point(x1, y1), new Point(x2, y2), colormap[obj.getPredictedClass()], 2, 0, 0);
            //Display label text
            labeltext = label + " " + String.format("%.2f", obj.getConfidence() * 100) + "%";
            int[] baseline = {0};
            Size textSize = getTextSize(labeltext, FONT_HERSHEY_DUPLEX, 1, 1, baseline);
            rectangle(mat, new Point(x1 + 2, y2 - 2), new Point(x1 + 2 + textSize.get(0), y2 - 2 - textSize.get(1)), colormap[obj.getPredictedClass()], FILLED, 0, 0);
            putText(mat, labeltext, new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, RGB(0, 0, 0));
        }
        return mat;
    }

}



