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

package ai.certifai.solution.object_detection;

import org.bytedeco.opencv.opencv_core.*;

import static org.bytedeco.opencv.global.opencv_core.flip;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

import org.bytedeco.javacv.*;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ColorConversionTransform;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.deeplearning4j.zoo.util.darknet.VOCLabels;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.event.KeyEvent;
import java.util.List;

/**
 * This example shows how to infer a TinyYOLOv2 Object Detection model trained on Pascal VOC dataset(20 classes).
 * The inference is done on your first camera, probably your webcam if you are using a laptop.
 * Change cameraNum to 1 if you wish to use your second camera.
 */

public class WebCamObjectDetection {
    //Camera position change between "front" and "back"
    //front camera requires flipping of the image
    private static String cameraPos = "front";

    //swap between camera with 0 -? on the parameter
    //Default is 0
    private static int cameraNum = 0;
    private static Thread thread;
    private static final int gridWidth = 13;
    private static final int gridHeight = 13;
    private static double detectionThreshold = 0.5;
    private static final int tinyyolowidth = 416;
    private static final int tinyyoloheight = 416;


    public static void main(String[] args) throws Exception {
        if (!cameraPos.equals("front") && !cameraPos.equals("back")) {
            throw new Exception("Unknown argument for camera position. Choose between front and back");
        }

        FrameGrabber grabber = FrameGrabber.createDefault(cameraNum);
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

        grabber.start();
        String winName = "Object Detection";
        CanvasFrame canvas = new CanvasFrame(winName);
        int w = grabber.getImageWidth();
        int h = grabber.getImageHeight();
        canvas.setCanvasSize(w, h);
        ZooModel model = TinyYOLO.builder().numClasses(0).build();
        ComputationGraph initializedModel = (ComputationGraph) model.initPretrained();
        NativeImageLoader loader = new NativeImageLoader(tinyyolowidth, tinyyoloheight, 3, new ColorConversionTransform(COLOR_BGR2RGB));
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        VOCLabels labels = new VOCLabels();

        while (true) {
            Frame frame = grabber.grab();

            //if a thread is null, create new thread
            if (thread == null) {
                thread = new Thread(() ->
                {
                    while (frame != null) {
                        try {
                            Mat rawImage = new Mat();
                            //Flip the camera if opening front camera
                            if (cameraPos.equals("front")) {
                                Mat inputImage = converter.convert(frame);
                                flip(inputImage, rawImage, 1);
                            } else {
                                rawImage = converter.convert(frame);
                            }
                            Mat resizeImage = new Mat();
                            resize(rawImage, resizeImage, new Size(tinyyolowidth, tinyyoloheight));
                            INDArray inputImage = loader.asMatrix(resizeImage);
                            scaler.transform(inputImage);
                            INDArray outputs = initializedModel.outputSingle(inputImage);
                            List<DetectedObject> objs = YoloUtils.getPredictedObjects(Nd4j.create(((TinyYOLO) model).getPriorBoxes()), outputs, detectionThreshold, 0.4);

                            for (DetectedObject obj : objs) {
                                double[] xy1 = obj.getTopLeftXY();
                                double[] xy2 = obj.getBottomRightXY();
                                String label = labels.getLabel(obj.getPredictedClass());
                                int x1 = (int) Math.round(w * xy1[0] / gridWidth);
                                int y1 = (int) Math.round(h * xy1[1] / gridHeight);
                                int x2 = (int) Math.round(w * xy2[0] / gridWidth);
                                int y2 = (int) Math.round(h * xy2[1] / gridHeight);
                                rectangle(rawImage, new Point(x1, y1), new Point(x2, y2), Scalar.RED, 2, 0, 0);
                                putText(rawImage, label, new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, Scalar.GREEN);
                            }
                            canvas.showImage(converter.convert(rawImage));
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    }
                });
                thread.start();
            }
            KeyEvent t = canvas.waitKey(33);
            if ((t != null) && (t.getKeyCode() == KeyEvent.VK_Q)) {
                break;
            }
        }
        canvas.dispose();
    }
}