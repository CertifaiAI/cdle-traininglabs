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

package ai.certifai.solution.object_detection.ActorsDetector.tinyyolo;

import ai.certifai.solution.object_detection.ActorsDetector.tinyyolo.dataHelpers.NonMaxSuppression;
import org.bytedeco.opencv.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import org.bytedeco.opencv.opencv_videoio.*;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_HEIGHT;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_WIDTH;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Real-time inference with trained model 'TLDetectorActors'
 */
public class WebcamInference {
    private static final Logger log = LoggerFactory.getLogger(WebcamInference.class);
    private static final OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage();
    private static ComputationGraph model;

    // trained model file
    private static String modelFilename = new File(".").getAbsolutePath() + "/generated-models/TinyYOLO_TLDetectorActors.zip";

    // parameters matching the pretrained TinyYOLO model
    private static int width = 416;
    private static int height = 416;
    private static int nChannels = 3;
    private static int gridWidth = 13;
    private static int gridHeight = 13;

    // labels - model's classes
    private static Object[] labels = new String[]{"Amber Heard", "Jason Momoa", "Patrick Wilson"};

    // minimal confident of the detected faces to be display
    private static double detectionThreshold = 0.1;

    public static void main(String[] args) throws Exception {

        if (new File(modelFilename).exists()) {
            log.info("Load model...");
            model = ModelSerializer.restoreComputationGraph(modelFilename);
        } else {
            log.info("Model not found.");
        }

        // invoke webcam - inferencing
        final AtomicReference<VideoCapture> capture = new AtomicReference<>(new VideoCapture());
        capture.get().set(CAP_PROP_FRAME_WIDTH, width);
        capture.get().set(CAP_PROP_FRAME_HEIGHT, height);

        if (!capture.get().open(0)) {
            log.error("Can not open the cam !!!");
        }

        Mat colorimg = new Mat();
        CanvasFrame mainframe = new CanvasFrame("Real-time Detector", CanvasFrame.getDefaultGamma() / 2.2);
        mainframe.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
        mainframe.setCanvasSize(width, height);
        mainframe.setLocationRelativeTo(null);
        mainframe.setVisible(true);

        while (true) {
            while (capture.get().read(colorimg) && mainframe.isVisible()) {
                long st = System.currentTimeMillis();
                resize(colorimg, colorimg, new Size(width, height));
                detect(colorimg, detectionThreshold);
                double per = (System.currentTimeMillis() - st) / 1000.0;
                log.info("It takes " + per + "Seconds to make detection");
                putText(colorimg, "Detection Time : " + per + " ms", new Point(10, 25), 2,.9, Scalar.YELLOW);

                mainframe.showImage(converter.convert(colorimg));
                try {
                    Thread.sleep(20);
                } catch (InterruptedException ex) {
                    log.error(ex.getMessage());
                }
            }
        }
    }

    public static void detect(Mat image, double threshold) {

        // access output layer
        org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout =
            (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) model.getOutputLayer(0);
        NativeImageLoader loader = new NativeImageLoader(width, height, nChannels);
        INDArray ds = null;
        try {
            ds = loader.asMatrix(image);
        } catch (IOException ex) {
            log.error(ex.getMessage());
        }
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(ds);
        INDArray results = model.outputSingle(ds);

        // filter predicted classes with confident higher than detection threshold
        List<DetectedObject> objs = yout.getPredictedObjects(results, threshold);

        // Non-max suppression is a way to eliminate points that do not lie in important edges (optional in this case)
        List<DetectedObject> objects = NonMaxSuppression.getObjects(objs);

        // draw boxes
        drawBoxes(image, objects);
    }

    private static void drawBoxes(Mat image, List<DetectedObject> objects) {
        for (DetectedObject obj : objects) {
            double[] xy1 = obj.getTopLeftXY();
            double[] xy2 = obj.getBottomRightXY();
            int predictedClass = obj.getPredictedClass();
            System.out.println("Predicted class " + labels[predictedClass]);
            int x1 = (int) Math.round(width * xy1[0] / gridWidth);
            int y1 = (int) Math.round(height * xy1[1] / gridHeight);
            int x2 = (int) Math.round(width * xy2[0] / gridWidth);
            int y2 = (int) Math.round(height * xy2[1] / gridHeight);
            rectangle(image, new Point(x1, y1), new Point(x2, y2), Scalar.RED);
            putText(image, (String) labels[predictedClass], new Point(x1 + 2, y2 - 2), 1, .8, Scalar.RED);
        }
    }
}
