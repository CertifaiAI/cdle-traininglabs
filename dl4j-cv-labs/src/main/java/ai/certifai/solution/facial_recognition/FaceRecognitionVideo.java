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

package ai.certifai.solution.facial_recognition;

import ai.certifai.solution.facial_recognition.detection.FaceDetector;
import ai.certifai.solution.facial_recognition.detection.FaceLocalization;
import ai.certifai.solution.facial_recognition.detection.OpenCV_DeepLearningFaceDetector;
import ai.certifai.solution.facial_recognition.detection.OpenCV_HaarCascadeFaceDetector;
import ai.certifai.solution.facial_recognition.identification.DistanceFaceIdentifier;
import ai.certifai.solution.facial_recognition.identification.FaceIdentifier;
import ai.certifai.solution.facial_recognition.identification.Prediction;
import ai.certifai.solution.facial_recognition.identification.feature.InceptionResNetFeatureProvider;
import ai.certifai.solution.facial_recognition.identification.feature.VGG16FeatureProvider;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;
import org.bytedeco.opencv.opencv_videoio.VideoWriter;
import org.nd4j.linalg.io.ClassPathResource;
import org.opencv.videoio.Videoio;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.flip;
import static org.bytedeco.opencv.global.opencv_highgui.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_HEIGHT;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_WIDTH;

/*
 *
 * FACE RECOGNITION USING YOUR OWN VIDEO
 * ------------------------------------------
 * This is an example of how to perform face recognition on video
 * The pipeline starts from loading video -> face detection -> face recognition
 * Face detection can be done using traditional CV (Haar Cascade) or Deep Learning (Single Shot Detector(SSD))
 * Face recognition is done by matching the input face with the face in the database that has the smallest distance
 *
 * Distance can be calculated either by Euclidean distance or Cosine Similarity
 * Face database is located in the "dl4j-cv-labs/src/main/resources/FaceDB"
 * User can add or remove faces and group them in a same folder and the folder name will act as the Label
 *
 *
 *
 * TODO
 * -------------------------------------------
 * 1.   Create a folder with your name (eg. Alex) under "dl4j-cv-labs/src/main/resources/FaceDB"
 * 2.   Place your face images under the folder you created
 * 3.   Place your video file under "dl4j-cv-labs/src/main/resources/FaceRecognition_input/video/"
 * 4.   Run this java file
 * 5.   You can find the output under "dl4j-cv-labs/src/main/resources/FaceRecognition_output/video/"
 *
 */

public class FaceRecognitionVideo {

    private static Logger log = LoggerFactory.getLogger(FaceRecognitionVideo.class);
    private static final int WIDTH = 1280;
    private static final int HEIGHT = 720;
    private static final String outputWindowsName = "Face Recognition in Processing......";
    private static final String inputPath = "dl4j-cv-labs/src/main/resources/FaceRecognition_input/video/";
    private static final String outputPath = "dl4j-cv-labs/src/main/resources/FaceRecognition_output/video/";

    public static void main(String[] args) throws IOException, ClassNotFoundException {

        // Remove placeholder.txt
        placeHolderRemover();

        // Read folder
        File folder = new File(inputPath);

        // Print out error message if there is no input video file inside input folder
        if (folder.listFiles().length < 1) {
            System.out.println("Please place your video under " + Paths.get(System.getProperty("user.dir"), inputPath));
        } else {
            // Select your FaceDetector
            FaceDetector FaceDetector = getFaceDetector(ai.certifai.solution.facial_recognition.detection.FaceDetector.OPENCV_DL_FACEDETECTOR);

            // Select your FaceIdentifier
            FaceIdentifier FaceIdentifier = getFaceIdentifier(ai.certifai.solution.facial_recognition.identification.FaceIdentifier.FEATURE_DISTANCE_VGG16_PREBUILT);

            // Loop all the video files inside the input folder and process them
            for (int i = 0; i < folder.listFiles().length; i++) {

                // Get filename
                String filename = folder.listFiles()[i].getName();

                // Load video using VideoCapture
                VideoCapture videoCapture = new VideoCapture(inputPath + filename);
                videoCapture.set(CAP_PROP_FRAME_WIDTH, WIDTH);
                videoCapture.set(CAP_PROP_FRAME_HEIGHT, HEIGHT);
                namedWindow(outputWindowsName, WINDOW_NORMAL);
                resizeWindow(outputWindowsName, 1280, 720);

                // Export video using VideoWriter
                Size frameSize = new Size((int) videoCapture.get(Videoio.CAP_PROP_FRAME_WIDTH), (int) videoCapture.get(Videoio.CAP_PROP_FRAME_HEIGHT));
                VideoWriter videoWriter = new VideoWriter(outputPath + "output_" + filename, VideoWriter.fourcc((byte) 'x', (byte) '2', (byte) '6', (byte) '4'), videoCapture.get(Videoio.CAP_PROP_FPS), frameSize, true);

                Mat image = new Mat();
                Mat cloneCopy = new Mat();

                while (videoCapture.read(image)) {
                    flip(image, image, 1);

                    // Perform face detection
                    image.copyTo(cloneCopy);
                    FaceDetector.detectFaces(cloneCopy);
                    List<FaceLocalization> faceLocalizations = FaceDetector.getFaceLocalization();
                    annotateFaces(faceLocalizations, image);

                    // Perform face recognition
                    image.copyTo(cloneCopy);
                    List<List<Prediction>> faceIdentities = FaceIdentifier.recognize(faceLocalizations, cloneCopy);
                    labelIndividual(faceIdentities, image);

                    // Display output in a window
                    imshow(outputWindowsName, image);

                    // Export the video
                    videoWriter.write(image);

                    char key = (char) waitKey(20);
                    // Exit this loop on escape
                    if (key == 27) {
                        destroyAllWindows();
                        break;
                    }
                }

                // Export the video
                videoCapture.release();
                videoWriter.release();
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
                return null;
        }
    }

    // Interface to change between different face recognition class
    // Modify values below to tweak the performance:
    //      *threshold: threshold to check if the detected face is within the database, do not label the face if lower than threshold
    //      *numSamples: the top n-number of samples that has the highest confidence that is use for averaging
    private static FaceIdentifier getFaceIdentifier(String faceIdentifier) throws IOException, ClassNotFoundException {
        switch (faceIdentifier) {
            case FaceIdentifier.FEATURE_DISTANCE_VGG16_PREBUILT:
                return new DistanceFaceIdentifier(
                        new VGG16FeatureProvider(),
                        new ClassPathResource("FaceDB").getFile(), 0.3, 3);
            case FaceIdentifier.FEATURE_DISTANCE_INCEPTION_RESNET_PREBUILT:
                return new DistanceFaceIdentifier(
                        new InceptionResNetFeatureProvider(),
                        new ClassPathResource("FaceDB").getFile(), 0.3, 3);
            default:
                return null;
        }
    }

    // Method to draw the predicted bounding box of the detected face
    private static void annotateFaces(List<FaceLocalization> faceLocalizations, Mat image) {
        for (FaceLocalization i : faceLocalizations) {
            rectangle(image, new Rect(new Point((int) i.getLeft_x(), (int) i.getLeft_y()), new Point((int) i.getRight_x(), (int) i.getRight_y())), new Scalar(0, 255, 0, 0), 2, 8, 0);
        }
    }

    // Method to label the predicted person's name
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

    // To commit empty folder to github, have to have file inside a folder, therefore this method is use to remove the placeholder.txt
    public static void placeHolderRemover() {
        File inputPlaceHolder = new File(inputPath + "placeholder.txt");
        File outputPlaceHolder = new File(outputPath + "placeholder.txt");

        if (inputPlaceHolder.exists()) {
            inputPlaceHolder.delete();
        }
        if (outputPlaceHolder.exists()) {
            outputPlaceHolder.delete();
        }
    }
}