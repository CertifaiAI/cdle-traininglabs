package global.skymind.training.facial_recognition;

import global.skymind.training.facial_recognition.detection.FaceDetector;
import global.skymind.training.facial_recognition.detection.FaceLocalization;
import global.skymind.training.facial_recognition.detection.OpenCV_DeepLearningFaceDetector;
import global.skymind.training.facial_recognition.detection.OpenCV_HaarCascadeFaceDetector;
import global.skymind.training.facial_recognition.identification.DistanceFaceIdentifier;
import global.skymind.training.facial_recognition.identification.FaceIdentifier;
import global.skymind.training.facial_recognition.identification.Prediction;
import global.skymind.training.facial_recognition.identification.feature.RamokFaceNetFeatureProvider;
import global.skymind.training.facial_recognition.identification.feature.VGG16FeatureProvider;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.flip;
import static org.bytedeco.opencv.global.opencv_highgui.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_HEIGHT;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_WIDTH;


public class FaceRecognitionWebcam {
    private static final Logger log = LoggerFactory.getLogger(FaceRecognitionWebcam.class);
    private static final int WIDTH = ;
    private static final int HEIGHT = ;
    private static final String outputWindowsName = "";

    public static void main(String[] args) throws Exception {

        //        Switch between FaceDetector and FaceIdentifier to test different capabilities
        FaceDetector FaceDetector = ;
        FaceIdentifier FaceIdentifier = ;

        VideoCapture capture = new VideoCapture();
        capture.set(CAP_PROP_FRAME_WIDTH, WIDTH);
        capture.set(CAP_PROP_FRAME_HEIGHT, HEIGHT);
        namedWindow(outputWindowsName, WINDOW_NORMAL);
        resizeWindow(outputWindowsName, 1280, 720);

        if (!capture.open(0)) {
            System.out.println("Can not open the camera !!!");
        }

        Mat image = new Mat();
        Mat cloneCopy = new Mat();

        while (capture.read(image)) {
            flip(image, image, 1);

            // Perform face detection
            image.copyTo(cloneCopy);
            FaceDetector.(cloneCopy);
            List<FaceLocalization> faceLocalizations = FaceDetector.();
            (faceLocalizations, image);

            // Perform face recognition
            image.copyTo(cloneCopy);
            List<List<Prediction>> faceIdentities = FaceIdentifier.(faceLocalizations, cloneCopy);
            (faceIdentities, image);

            (outputWindowsName, image);

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

    private static FaceIdentifier getFaceIdentifier(String faceIdentifier) throws IOException, ClassNotFoundException {
        switch (faceIdentifier) {
            case FaceIdentifier.FEATURE_DISTANCE_VGG16_PREBUILT:
                return new DistanceFaceIdentifier(
                        new VGG16FeatureProvider(),
                        new ClassPathResource("FaceDB").getFile(), 1, 0.3, 3);
            case FaceIdentifier.FEATURE_DISTANCE_RAMOK_FACENET_PREBUILT:
                return new DistanceFaceIdentifier(
                        new RamokFaceNetFeatureProvider(),
                        new ClassPathResource("FaceDB").getFile(), 1, 0.3, 3);
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
