/*
 * Copyright (c) 2020-2021 CertifAI Sdn. Bhd.
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
 *
 */

package ai.certifai.solution.object_detection.MetalDefectsDetector;

import ai.certifai.Helper;
import org.apache.commons.io.FileUtils;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.CV_8U;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.helper.opencv_core.RGB;

/**
 * This is an example of a metal surface defect detection using YOLOv2 architecture.
 * This example download a pretrained weights and performs inference on test datasets.
 * NOTE: DUE TO THE MEMORY CONSTRAINT, THE MODEL NEEDS TO BE TRAINED USING CPU
 */

public class MetalSurfaceDefectDetector_InferenceUsingTrainedModel {

    private static final Logger log = LoggerFactory.getLogger(MetalSurfaceDefectDetector_InferenceUsingTrainedModel.class);
    private static int seed = 123;
    private static double detectionThreshold = 0.5;
    private static int batchSize = 8;
    private static List<String> labels;
    private static File trainedModel = new File(System.getProperty("user.dir"), "generated-models/MetalSurfaceDefects_yolov2_trained.zip");
    private static ComputationGraph model;
    private static final Scalar BLUE = RGB(0, 0, 255);
    private static final Scalar GREEN = RGB(0, 255, 0);
    private static final Scalar RED = RGB(255, 0, 0);
    public static final Scalar YELLOW = RGB(255, 225, 0);
    private static final Scalar PINK = RGB(255, 0, 225);
    private static final Scalar CYAN = RGB(0, 225, 225);
    private static Scalar[] colormap = {BLUE, GREEN, RED, YELLOW, PINK, CYAN};
    private static String labeltext = null;

    public static void main(String[] args) throws Exception {
        MetalDefectDataSetIterator.setup();
        downloadModel();

        //        STEP 1 : Create iterators
        RecordReaderDataSetIterator trainIter = MetalDefectDataSetIterator.trainIterator(batchSize);
        RecordReaderDataSetIterator testIter = MetalDefectDataSetIterator.testIterator(1);

        labels = trainIter.getLabels();

        //      STEP 2: Load model
        Nd4j.getRandom().setSeed(seed);
        log.info("Load model...");
        model = ModelSerializer.restoreComputationGraph(trainedModel);

        //      STEP 3: model inference
        OfflineValidationWithTestDataset(testIter);
    }

    private static void downloadModel() throws IOException {
        String remoteUrl = Helper.getPropValues("models.metaldefectsdetector.url");
        if(!trainedModel.exists() || !Helper.getCheckSum(trainedModel.getAbsolutePath())
                .equalsIgnoreCase(Helper.getPropValues("models.metaldefectsdetector.hash"))){
            log.info("Downloading model to " + trainedModel.toString());
            FileUtils.copyURLToFile(new URL(remoteUrl), trainedModel);
        }
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
