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

package ai.certifai.training.facial_recognition;

import ai.certifai.training.facial_recognition.detection.FaceDetector;
import ai.certifai.training.facial_recognition.detection.FaceLocalization;
import ai.certifai.training.facial_recognition.detection.OpenCV_DeepLearningFaceDetector;
import ai.certifai.training.facial_recognition.detection.OpenCV_HaarCascadeFaceDetector;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;

import static ai.certifai.training.facial_recognition.detection.FaceDetector.OPENCV_DL_FACEDETECTOR;
import static ai.certifai.training.facial_recognition.detection.FaceDetector.OPENCV_HAAR_CASCADE_FACEDETECTOR;
import static org.bytedeco.opencv.global.opencv_core.flip;
import static org.bytedeco.opencv.global.opencv_highgui.*;
import static org.bytedeco.opencv.global.opencv_highgui.destroyAllWindows;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_HEIGHT;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_WIDTH;

public class FaceAnonymizationWebCam {
    private static final Logger log = LoggerFactory.getLogger(FaceAnonymizationWebCam.class);
    private static final int WIDTH = 1280;
    private static final int HEIGHT = 720;
    private static final String outputWindowsName = "Face Recognition Example - DL4J";

    public static void main(String[] args) throws Exception {
        /**
         * STEP 1 : Select your face detector and face identifier
         * You can switch between different FaceDetector and FaceIdentifier options to test its performance
         *
         * - COMPLETE THE FOLLOWING LINE OF CODE (LINE 62)
         * */
//        FaceDetector FaceDetector = getFaceDetector();


        VideoCapture capture = new VideoCapture();
        capture.set(CAP_PROP_FRAME_WIDTH, WIDTH);
        capture.set(CAP_PROP_FRAME_HEIGHT, HEIGHT);
        namedWindow(outputWindowsName, WINDOW_NORMAL);
        resizeWindow(outputWindowsName, 1280, 720);

        if (!capture.open(0)) {
            System.out.println("Cannot open the camera !!!");
        }

        Mat image = new Mat();
        Mat cloneCopy = new Mat();

        while (capture.read(image)) {
            flip(image, image, 1);


            /**
             *  STEP 2 : Perform face detection
             * UNCOMMENT THE FOLLOWING PIECE OF CODE (LINE 87 - 90)
             * **/
//            image.copyTo(cloneCopy);
//            FaceDetector.detectFaces(cloneCopy);
//            List<FaceLocalization> faceLocalizations = FaceDetector.getFaceLocalization();
//            annotateFaces(faceLocalizations, image);

            imshow(outputWindowsName, image);

            char key = (char) waitKey(20);
            // Exit this loop on escape:
            if (key == 27) {
                destroyAllWindows();
                break;
            }
        }
    }

    private static FaceDetector getFaceDetector(String faceDetector) throws IOException {
        switch (faceDetector) {
            case OPENCV_HAAR_CASCADE_FACEDETECTOR:
                return new OpenCV_HaarCascadeFaceDetector();
            case OPENCV_DL_FACEDETECTOR:
                return new OpenCV_DeepLearningFaceDetector(300, 300, 0.8);
            default:
                return  null;
        }
    }



    //    Method to draw the predicted bounding box of the detected face
    private static void annotateFaces(List<FaceLocalization> faceLocalizations, Mat image) {

        for (FaceLocalization i : faceLocalizations){
            Rect roi = new Rect(new Point((int) i.getLeft_x(),(int) i.getLeft_y()), new Point((int) i.getRight_x(),(int) i.getRight_y()));


            /***
             * STEP 3: Add Gaussian blur here
            * ENTER YOUR CODE FOR HERE
            * **/

            rectangle(image, roi, new Scalar(255, 255, 255, 0),0,8,0);
        }
    }



}
