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

package ai.certifai.training.facial_recognition;

import ai.certifai.solution.facial_recognition.detection.FaceDetector;
import ai.certifai.solution.facial_recognition.detection.FaceLocalization;
import ai.certifai.solution.facial_recognition.detection.OpenCV_DeepLearningFaceDetector;
import ai.certifai.solution.facial_recognition.detection.OpenCV_HaarCascadeFaceDetector;
import ai.certifai.solution.facial_recognition.identification.DistanceFaceIdentifier;
import ai.certifai.solution.facial_recognition.identification.FaceIdentifier;
import ai.certifai.solution.facial_recognition.identification.Prediction;
import ai.certifai.solution.facial_recognition.identification.feature.RamokFaceNetFeatureProvider;
import ai.certifai.solution.facial_recognition.identification.feature.VGG16FeatureProvider;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_highgui.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

/**
 * This is an example of how to perform face recognition on images
 * The pipeline starts from loading image -> face detection -> face recognition
 * Face detection can be done using traditional CV ( Haar Cascade ) or Deep Learning ( Single Shot Detector(SSD) )
 * Face recognition is done by matching the input face with the face in the database that has the smallest distance
 *
 * Distance can be calculated either by Euclidean distance or Cosine Similarity
 * Face database is located in the resource folder "FaceDB"
 * User can add or remove faces and group them in a same folder and the folder name will act as the Label
 *
 * TODO
 *
 *  1.  Inside the resources/FaceRecognition_input/img folder, create a folder and name the folder with your name,
 *      eg: KengHooi, store the images of your face inside the folder.
 *  2.  Upload your test image to resources/FaceRecognition_input/img folder
 *  3.  Using the INPUT_PATH to load all the files inside the input folder in a File array.
 *  4.  Use a List of Mat to store the images.
 *  5.  By looping through each file , perform following actions:
 *          a.  Read the image file.
 *          b.  Make a copy of the Mat image object.
 *          c.  Use the copy to perform face detection and get the height & width of the face.
 *          d.  Draw boundary box around the detected face by using the annotateFaces method.
 *          e.  Make another copy of the Mat image and perform face identification, calculate the distance between faces
 *              to get similarity score and label them by using labelIndividual method.
 *          f.  Save the labeled image.
 *          g.  Display the result of the face recognition.
 *          
 **/

public class FaceRecognitionImage {
    public static final String INPUT_PATH = System.getProperty("user.dir") + "/dl4j-cv-labs/src/main/resources/FaceRecognition_input/img/";
    public static final String OUTPUT_PATH = System.getProperty("user.dir") + "/dl4j-cv-labs/src/main/resources/FaceRecognition_output/img/";

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        //  STEP 1 : Store your images inside FaceDB
        //  STEP 2 : Upload test image to FaceRecognition_input/img

        //  Loading all files in the directory folder
        //  STEP 3 : ENTER YOUR CODE HERE

        FaceDetector FaceDetector = getFaceDetector(ai.certifai.solution.facial_recognition.detection.FaceDetector.OPENCV_DL_FACEDETECTOR);
        FaceIdentifier FaceIdentifier = getFaceIdentifier(ai.certifai.solution.facial_recognition.identification.FaceIdentifier.FEATURE_DISTANCE_VGG16_PREBUILT);

        //  Using List to store a list of image
        //  STEP 4 :  ENTER YOUR CODE HERE

        //  Looping through all files in the folder
//        for () {
//            //  Assigning the absolute path of each file to imgPath
//            String imgPath = INPUT_PATH + // YOUR FILE NAME
//
//            //  Read image and store it in the List of Mat
//            //  STEP 5a : ENTER YOUR CODE HERE
//
//            //  Make a copy of the image for face detection
//            //  STEP 5b : ENTER YOUR CODE HERE
//
//            //  STEP 5c : Perform face detection and get the height & width of the face
//            FaceDetector;
//            List<FaceLocalization> faceLocalizations;
//
//            //  STEP 5d : Draw boundary box
//            annotateFaces();
//
//            //  STEP 5e : Perform face identification, calculate the distance between faces to get similarity score and label them
//            List<List<Prediction>> faceIdentities ;
//            labelIndividual();
//
//            //  STEP 5f : Saving file in the resources/FaceRecognition_output/img folder
//            imwrite(OUTPUT_PATH + "output_" + // File Name );
//
//            //  STEP 5g :  Displaying output of the face recognition
//            imshow();
//        }
        // Press Esc to close all windows
        if (waitKey(0) == 27) {
            destroyAllWindows();
        }
    }

    private static FaceDetector getFaceDetector(String faceDetector) throws IOException {
        switch (faceDetector) {
            case FaceDetector.OPENCV_HAAR_CASCADE_FACEDETECTOR:
                return new OpenCV_HaarCascadeFaceDetector();
            case FaceDetector.OPENCV_DL_FACEDETECTOR:
                return new OpenCV_DeepLearningFaceDetector(300, 300, 0.8);
            default:
                return null;
        }
    }

    //        Interface to change between different face recognition class
    //        Modify values below to tweak the performance
    //          *threshold: threshold to check if the detected face is within the database, do not label the face if lower than threshold
    //          *numSamples: the top n-number of samples that has the highest confidence that is use for averaging

    private static FaceIdentifier getFaceIdentifier(String faceIdentifier) throws
            IOException, ClassNotFoundException {
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
        for (FaceLocalization i : faceLocalizations) {
            rectangle(image, new Rect(new Point((int) i.getLeft_x(), (int) i.getLeft_y()), new Point((int) i.getRight_x(), (int) i.getRight_y())), new Scalar(0, 255, 0, 0), 2, 8, 0);
        }
    }

    //    Method to label the predicted person's name
    private static void labelIndividual(List<List<Prediction>> faceIdentities, Mat image) {
        for (List<Prediction> i : faceIdentities) {
            for (int j = 0; j < i.size(); j++) {
                putText(
                        image,
                        i.get(j).toString(),
                        new Point(
                                (int) i.get(j).getFaceLocalization().getLeft_x() + 2,
                                (int) i.get(j).getFaceLocalization().getLeft_y() - 5
                        ),
                        FONT_HERSHEY_COMPLEX,
                        0.5,
                        Scalar.YELLOW
                );
            }
        }
    }
}
