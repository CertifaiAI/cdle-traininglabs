package ai.certifai.solution.facial_recognition;

import ai.certifai.solution.facial_recognition.detection.FaceDetector;
import ai.certifai.solution.facial_recognition.detection.FaceLocalization;
import ai.certifai.solution.facial_recognition.detection.OpenCV_DeepLearningFaceDetector;
import ai.certifai.solution.facial_recognition.detection.OpenCV_HaarCascadeFaceDetector;
import ai.certifai.training.image_processing.Display;
import org.bytedeco.opencv.opencv_core.*;

import java.io.File;
import java.io.IOException;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class FaceDetectCrop {
    private static final String inputPath = "dl4j-cv-labs/src/main/resources/FaceCrop/input/";
    private static final String outputPath = "dl4j-cv-labs/src/main/resources/FaceCrop/output/";

    public static void main(String[] args) throws IOException {
        // Read folder
        File folder = new File(inputPath);

        // Print out error message if there is no input image file inside input folder
        if (folder.listFiles().length < 1) {
            System.out.println("Please place your images under the input folder!");
        } else {
            FaceDetector FaceDetector = getFaceDetector(ai.certifai.solution.facial_recognition.detection.FaceDetector.OPENCV_DL_FACEDETECTOR);
            for (int i = 0; i < folder.listFiles().length; i++) {
                String filename = folder.listFiles()[i].getName();
                Mat rawImg = imread(inputPath + filename);
                Display.display(rawImg, "Input_Img " + (i + 1));

                // assuming detectFaces() will shrink img, that's why need cloneCopy
                Mat cloneCopy = new Mat();
                rawImg.copyTo(cloneCopy);
                FaceDetector.detectFaces(cloneCopy);
                List<FaceLocalization> faceLocalizations = FaceDetector.getFaceLocalization();
                Mat cropImg = crop(faceLocalizations, rawImg);
                resize(cropImg, cropImg, new Size(224, 224));

                Display.display(cropImg, "Cropped_Img " + (i + 1));
                imwrite(outputPath + filename, cropImg);
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

    // Crop the detected face
    private static Mat crop(List<FaceLocalization> faceLocalizations, Mat image) {
        Rect rect = null;
        for (FaceLocalization i : faceLocalizations) {
            rect = new Rect(new Point((int) i.getLeft_x(), (int) i.getLeft_y()), new Point((int) i.getRight_x(), (int) i.getRight_y()));
        }
        Mat img_roi = new Mat(image, rect);
        return img_roi;
    }
}
