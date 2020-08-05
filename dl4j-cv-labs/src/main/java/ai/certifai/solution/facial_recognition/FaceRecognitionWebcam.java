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

package ai.certifai.solution.facial_recognition;

import ai.certifai.solution.facial_recognition.detection.FaceDetector;
import ai.certifai.solution.facial_recognition.detection.FaceLocalization;
import ai.certifai.solution.facial_recognition.detection.OpenCV_DeepLearningFaceDetector;
import ai.certifai.solution.facial_recognition.detection.OpenCV_HaarCascadeFaceDetector;
import ai.certifai.solution.facial_recognition.identification.DistanceFaceIdentifier;
import ai.certifai.solution.facial_recognition.identification.FaceIdentifier;
import ai.certifai.solution.facial_recognition.identification.Prediction;
import ai.certifai.solution.facial_recognition.identification.feature.RamokFaceNetFeatureProvider;
import ai.certifai.solution.facial_recognition.identification.feature.VGG16FeatureProvider;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.bytedeco.opencv.global.opencv_core.flip;
import static org.bytedeco.opencv.global.opencv_highgui.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_HEIGHT;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_WIDTH;

import java.io.IOException;
import java.util.List;
/**
 * This is an example of a simple face recognition pipeline.
 * The pipeline starts from video streaming -> face detection -> face recognition
 * Face detection can be done using traditional CV ( Haar Cascade ) or Deep Learning ( Single Shot Detector(SSD) )
 * Face recognition is done by matching the input face with the face in the database that has the smallest distance
 *
 * Distance can be calculated either by Euclidean distance or Cosine Similarity
 * Face database is located in the resource folder "FaceDB"
 * User can add or remove faces and group them in a same folder and the folder name will act as the Label
 * **/

public class FaceRecognitionWebcam {
    private static final Logger log = LoggerFactory.getLogger(FaceRecognitionWebcam.class);
    private static final int WIDTH = 1280;
    private static final int HEIGHT = 720;
    private static final String outputWindowsName = "Face Recognition Example - DL4J";

    public static void main(String[] args) throws Exception {
        //        STEP 1 : Select your face detector and face identifier
        //        You can switch between different FaceDetector and FaceIdentifier options to test its performance
        FaceDetector FaceDetector = getFaceDetector(ai.certifai.solution.facial_recognition.detection.FaceDetector.OPENCV_HAAR_CASCADE_FACEDETECTOR);
        FaceIdentifier FaceIdentifier = getFaceIdentifier(ai.certifai.solution.facial_recognition.identification.FaceIdentifier.FEATURE_DISTANCE_RAMOK_FACENET_PREBUILT);

        //        STEP 2 : Stream the video frame from camera
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

            //        STEP 3 : Perform face detection
            image.copyTo(cloneCopy);
            FaceDetector.detectFaces(cloneCopy);
            List<FaceLocalization> faceLocalizations = FaceDetector.getFaceLocalization();
            annotateFaces(faceLocalizations, image);

            //        STEP 4 : Perform face recognition
            image.copyTo(cloneCopy);
            List<List<Prediction>> faceIdentities = FaceIdentifier.recognize(faceLocalizations, cloneCopy);
            labelIndividual(faceIdentities, image);

            //        STEP 5 : Display output in a window
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
            case FaceDetector.OPENCV_HAAR_CASCADE_FACEDETECTOR:
                return new OpenCV_HaarCascadeFaceDetector();
            case FaceDetector.OPENCV_DL_FACEDETECTOR:
                return new OpenCV_DeepLearningFaceDetector(300, 300, 0.8);
            default:
                return  null;
        }
    }

    //        Interface to change between different face recognition class
    //        Modify values below to tweak the performance
    //          *threshold: threshold to check if the detected face is within the database, do not label the face if lower than threshold
    //          *numSamples: the top n-number of samples that has the highest confidence that is use for averaging

    private static FaceIdentifier getFaceIdentifier(String faceIdentifier) throws IOException, ClassNotFoundException {
        switch (faceIdentifier) {
            case FaceIdentifier.FEATURE_DISTANCE_VGG16_PREBUILT:
                return new DistanceFaceIdentifier(
                        new VGG16FeatureProvider(),
                        new ClassPathResource("FaceDB").getFile(), 0.3, 3);
            case FaceIdentifier.FEATURE_DISTANCE_RAMOK_FACENET_PREBUILT:
                return new DistanceFaceIdentifier(
                        new RamokFaceNetFeatureProvider(),
                        new ClassPathResource("FaceDB").getFile(), 0.3, 3);
            default:
                return null;
        }
    }

    //    Method to draw the predicted bounding box of the detected face
    private static void annotateFaces(List<FaceLocalization> faceLocalizations, Mat image) {
        for (FaceLocalization i : faceLocalizations){
            rectangle(image,new Rect(new Point((int) i.getLeft_x(),(int) i.getLeft_y()), new Point((int) i.getRight_x(),(int) i.getRight_y())), new Scalar(0, 255, 0, 0),2,8,0);
        }
    }
    //    Method to label the predicted person's name
    private static void labelIndividual(List<List<Prediction>> faceIdentities, Mat image) {
        for (List<Prediction> i: faceIdentities){
            for(int j=0; j<i.size(); j++)
            {
                putText(
                        image,
                        i.get(j).toString(),
                        new Point(
                                (int)i.get(j).getFaceLocalization().getLeft_x() + 2,
                                (int)i.get(j).getFaceLocalization().getLeft_y() - 5
                        ),
                        FONT_HERSHEY_COMPLEX,
                        0.5,
                        Scalar.YELLOW
                );
            }
        }
    }
}
