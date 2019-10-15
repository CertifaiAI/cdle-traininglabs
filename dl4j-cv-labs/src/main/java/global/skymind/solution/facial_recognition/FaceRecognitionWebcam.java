package global.skymind.solution.facial_recognition;

import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.*;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.bytedeco.opencv.global.opencv_core.flip;
import static org.bytedeco.opencv.global.opencv_highgui.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;


public class FaceRecognitionWebcam {
    private static final Logger log = LoggerFactory.getLogger(FaceRecognitionWebcam.class);
    private static Frame frame = null;
    private static Mat gray = new Mat();
    private static Mat rawImageClone = new Mat();
    private static Mat face = new Mat();
    private static Mat rawImage = new Mat();

    public static void main(String[] args) throws Exception {
        String fdmodel_path = new ClassPathResource("fdmodel/haarcascade_frontalface_default.xml").getFile().toString();
        CascadeClassifier face_cascade = new CascadeClassifier(fdmodel_path);
        doInference(face_cascade);
    }

    // Stream video frames from Webcam and get face detection
    private static void doInference(CascadeClassifier cascade) {

        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

        String cameraPos = "front";
        int cameraNum = 0;

        if (!cameraPos.equals("front") && !cameraPos.equals("back")) {
            try {
                throw new Exception("Unknown argument for camera position. Choose between front and back");
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        FrameGrabber grabber = null;
        try {
            grabber = FrameGrabber.createDefault(cameraNum);
        } catch (FrameGrabber.Exception e) {
            e.printStackTrace();
        }

        try {
            grabber.start();
        } catch (FrameGrabber.Exception e) {
            e.printStackTrace();
        }
        //Change the minimum and maximum sizes for face detection
        Size minSize = new Size(100, 100);
        Size maxSize = new Size(1000, 1000);

        while (true) {
            try {
                frame = grabber.grab();
            } catch (FrameGrabber.Exception e) {
                e.printStackTrace();
            }

            //Flip the camera if opening front camera
            if (cameraPos.equals("front")) {
                Mat inputImage = converter.convert(frame);
                flip(inputImage, rawImage, 1);
            } else {
                rawImage = converter.convert(frame);
            }

            cvtColor(rawImage, gray, COLOR_BGR2GRAY);
            rawImageClone = rawImage.clone();
            RectVector faces = new RectVector();
            cascade.detectMultiScale(gray, faces, 1.3, 5, 0, minSize, maxSize);

            for (int i = 0; i < faces.size(); i++) {
                Rect face_i = faces.get(i);
                rectangle(rawImage, face_i, new Scalar(255, 0, 0, 1), 2, 8, 0);
                face = new Mat(rawImageClone, face_i);
                imshow("Cropped Face", face);
            }
            imshow("Original", rawImage);

            char key = (char) waitKey(20);
            // Exit this loop on escape:
            if (key == 27) {
                destroyAllWindows();
                break;
            }
        }
    }
}


