package global.skymind.training.facial_recognition.detection;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY;

public class OpenCV_HaarCascadeFaceDetector extends FaceDetector {

    private static final Logger log = LoggerFactory.getLogger(OpenCV_HaarCascadeFaceDetector.class);
    private CascadeClassifier haar_cascade;
    private static Mat gray = new Mat();
    private static Size minSize = new Size(100, 100);
    private static Size maxSize = new Size(1000, 1000);
    private static RectVector faces = new RectVector();

    public OpenCV_HaarCascadeFaceDetector() throws IOException{
        setModel();
    }

    private void setModel() throws IOException{

        String model_path = new ClassPathResource("fdmodel/OpenCVHaarCascadeFaceDetector/haarcascade_frontalface_default.xml").getFile().toString();
        CascadeClassifier face_cascade = new CascadeClassifier(model_path);
        this.haar_cascade = face_cascade;
    }

    @Override
    public void detectFaces(Mat image) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
        haar_cascade.detectMultiScale(gray, faces, 1.3, 5, 0, minSize, maxSize);
    }

    @Override
    public List<FaceLocalization> getFaceLocalization() {
        List<FaceLocalization> faceLocalizations = new ArrayList();
        for (int i = 0; i < faces.size(); i++) {
            Rect face_i = faces.get(i);
            float tx = face_i.x();
            float ty = face_i.y();
            float bx = tx + face_i.width();
            float by = ty + face_i.height();

            faceLocalizations.add(new FaceLocalization(tx, ty, bx, by));
        }
        return faceLocalizations;
    }
}
