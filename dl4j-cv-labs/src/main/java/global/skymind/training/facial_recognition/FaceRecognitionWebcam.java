//package global.skymind.training.facial_recognition;
//
//import global.skymind.training.facial_recognition.detection.FaceDetector;
//import global.skymind.training.facial_recognition.detection.FaceLocalization;
//import global.skymind.training.facial_recognition.detection.OpenCV_DeepLearningFaceDetector;
//import global.skymind.training.facial_recognition.detection.OpenCV_HaarCascadeFaceDetector;
//import global.skymind.training.facial_recognition.identification.DistanceFaceIdentifier;
//import global.skymind.training.facial_recognition.identification.FaceIdentifier;
//import global.skymind.training.facial_recognition.identification.Prediction;
//import global.skymind.training.facial_recognition.identification.feature.RamokFaceNetFeatureProvider;
//import global.skymind.training.facial_recognition.identification.feature.VGG16FeatureProvider;
//import org.bytedeco.opencv.opencv_core.*;
//import org.bytedeco.opencv.opencv_videoio.VideoCapture;
//import org.nd4j.linalg.io.ClassPathResource;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;
//
//import static global.skymind.training.facial_recognition.detection.FaceDetector.OPENCV_DL_FACEDETECTOR;
//import static global.skymind.training.facial_recognition.detection.FaceDetector.OPENCV_HAAR_CASCADE_FACEDETECTOR;
//import static global.skymind.training.facial_recognition.identification.FaceIdentifier.FEATURE_DISTANCE_RAMOK_FACENET_PREBUILT;
//import static global.skymind.training.facial_recognition.identification.FaceIdentifier.FEATURE_DISTANCE_VGG16_PREBUILT;
//import static org.bytedeco.opencv.global.opencv_core.flip;
//import static org.bytedeco.opencv.global.opencv_highgui.*;
//import static org.bytedeco.opencv.global.opencv_imgproc.*;
//import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_HEIGHT;
//import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_WIDTH;
//
//import java.io.IOException;
//import java.util.List;
///**
// * This is an example of a simple face recognition pipeline.
// * The pipeline starts from video streaming -> face detection -> face recognition
// * Face detection can be done using traditional CV (Haar Cascade) or Deep Learning (SSD)
// * Face recognition is done by matching the input face with the face in the database that has the smallest distance
// *
// * Distance can be calculated either by Euclidean distance or Cosine Similarity
// * Face database is located in the resource folder "FaceDB"
// * User can add or remove faces and group them in a same folder and the folder name will act as the Label
// * **/
//
//public class FaceRecognitionWebcam {
//    private static final Logger log = LoggerFactory.getLogger(FaceRecognitionWebcam.class);
//    private static final int WIDTH = ;
//    private static final int HEIGHT = ;
//    private static final String outputWindowsName = "";
//
//    public static void main(String[] args) throws Exception {
//        //        STEP 1 : Select your face detector and face identifier
//        //        You can switch between different FaceDetector and FaceIdentifier options to test its performance
//        FaceDetector FaceDetector =
//        FaceIdentifier FaceIdentifier =
//
//        //        STEP 2 : Stream the video frame from camera
//        VideoCapture capture = new VideoCapture();
//        capture.set(CAP_PROP_FRAME_WIDTH, WIDTH);
//        capture.set(CAP_PROP_FRAME_HEIGHT, HEIGHT);
//        namedWindow(outputWindowsName, WINDOW_NORMAL);
//        resizeWindow(outputWindowsName, 1280, 720);
//
//        if (!capture.open(0)) {
//            System.out.println("Cannot open the camera !!!");
//        }
//
//        Mat image = new Mat();
//        Mat cloneCopy = new Mat();
//
//        while (capture.read(image)) {
//            flip(image, image, 1);
//
//            //        STEP 3 : Perform face detection
//            image.copyTo(cloneCopy);
//
//
//            //        STEP 4 : Perform face recognition
//            image.copyTo(cloneCopy);
//
//
//            //        STEP 5 : Display output in a window
//            imshow(outputWindowsName, image);
//
//            char key = (char) waitKey(20);
//            // Exit this loop on escape:
//            if (key == 27) {
//                destroyAllWindows();
//                break;
//            }
//        }
//    }
//
//    private static FaceDetector getFaceDetector(String faceDetector) throws IOException {
//        switch (faceDetector) {
//            case OPENCV_HAAR_CASCADE_FACEDETECTOR:
//                return new OpenCV_HaarCascadeFaceDetector();
//            case OPENCV_DL_FACEDETECTOR:
//                return new OpenCV_DeepLearningFaceDetector(300, 300, 0.8);
//            default:
//                return  null;
//        }
//    }
//
//    //        Interface to change between different face recognition class
//    //        Modify values below to tweak the performance
//    //          *numPredictions: number of face to predict in an inference
//    //          *threshold: threshold to check if the face detected is in the database
//    //          *numSamples: the top n-number of samples that has the highest confidence
//
//    private static FaceIdentifier getFaceIdentifier(String faceIdentifier) throws IOException, ClassNotFoundException {
//        switch (faceIdentifier) {
//            case FaceIdentifier.FEATURE_DISTANCE_VGG16_PREBUILT:
//                return new DistanceFaceIdentifier(
//                        new (),
//                        new ClassPathResource("").getFile(), );
//            case FaceIdentifier.FEATURE_DISTANCE_RAMOK_FACENET_PREBUILT:
//                return new DistanceFaceIdentifier(
//                        new (),
//                        new ClassPathResource("").getFile(), );
//            default:
//                return null;
//        }
//    }
//
//    //    Method to draw the predicted bounding box of the detected face
//    private static void annotateFaces(List<FaceLocalization> faceLocalizations, Mat image) {
//        for (FaceLocalization i : faceLocalizations){
//            rectangle(image,new Rect(new Point((int) i.getLeft_x(),(int) i.getLeft_y()), new Point((int) i.getRight_x(),(int) i.getRight_y())), new Scalar(0, 255, 0, 0),2,8,0);
//        }
//    }
//    //    Method to label the predicted person's name
//    private static void labelIndividual(List<List<Prediction>> faceIdentities, Mat image) {
//        for (List<Prediction> i: faceIdentities){
//            for(int j=0; j<i.size(); j++)
//            {
//                putText(
//                        image,
//                        i.get(j).toString(),
//                        new Point(
//                                (int)i.get(j).getFaceLocalization().getLeft_x() + 2,
//                                (int)i.get(j).getFaceLocalization().getLeft_y() - 5
//                        ),
//                        FONT_HERSHEY_COMPLEX,
//                        0.5,
//                        Scalar.YELLOW
//                );
//            }
//        }
//    }
//}
