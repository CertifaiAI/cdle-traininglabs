package global.skymind.solution.facial_recognition;

import global.skymind.solution.facial_recognition.detection.FaceDetector;
import global.skymind.solution.facial_recognition.detection.FaceLocalization;
import global.skymind.solution.facial_recognition.detection.OpenCV_DeepLearningFaceDetector;
import global.skymind.solution.facial_recognition.detection.OpenCV_HaarCascadeFaceDetector;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static global.skymind.solution.facial_recognition.detection.FaceDetector.OPENCV_DL_FACEDETECTOR;
import static global.skymind.solution.facial_recognition.detection.FaceDetector.OPENCV_HAAR_CASCADE_FACEDETECTOR;
import static org.bytedeco.opencv.global.opencv_core.flip;
import static org.bytedeco.opencv.global.opencv_highgui.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_HEIGHT;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_WIDTH;

import java.io.IOException;
import java.util.List;


public class FaceRecognitionWebcam {
    private static final Logger log = LoggerFactory.getLogger(FaceRecognitionWebcam.class);
    private static final int WIDTH = 1280;
    private static final int HEIGHT = 720;
    private static final String outputWindowsName = "Face Recognition Example - DL4J";

    public static void main(String[] args) throws Exception {
        FaceDetector FaceDetector = getFaceDetector(OPENCV_HAAR_CASCADE_FACEDETECTOR);
//        FaceDetector FaceDetector = getFaceDetector(OPENCV_DL_FACEDETECTOR);//        FaceIdentifier FaceIdentifier = getFaceIdentifier(com.skymindglobal.faceverification.identification.FaceIdentifier.FEATURE_DISTANCE_RAMOK_FACENET_PREBUILT);

        VideoCapture capture = new VideoCapture();
        capture.set(CAP_PROP_FRAME_WIDTH, WIDTH);
        capture.set(CAP_PROP_FRAME_HEIGHT, HEIGHT);
        namedWindow(outputWindowsName, WINDOW_NORMAL);
        resizeWindow(outputWindowsName, 1280, 720);

        if (!capture.open(0)) {
            System.out.println("Can not open the camera !!!");
        }

        Mat image = new Mat();

        while (capture.read(image)) {
            Mat cloneCopy = new Mat();

            // face detection
            image.copyTo(cloneCopy);

            FaceDetector.detectFaces(cloneCopy);
            List<FaceLocalization> faceLocalizations = FaceDetector.getFaceLocalization();
            annotateFaces(faceLocalizations, image);

            // face identification
            //                image.copyTo(cloneCopy);
            //                List<List<Prediction>> faceIdentities = FaceIdentifier.identify(faceLocalizations, cloneCopy);
            //                labelIndividual(faceIdentities, image);

            flip(image, image, 1);
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

    private static void annotateFaces(List<FaceLocalization> faceLocalizations, Mat image) {
        for (FaceLocalization i : faceLocalizations){
            rectangle(image,new Rect(new Point((int) i.getLeft_x(),(int) i.getLeft_y()), new Point((int) i.getRight_x(),(int) i.getRight_y())), new Scalar(0, 255, 0, 0),2,8,0);
        }
    }
}
