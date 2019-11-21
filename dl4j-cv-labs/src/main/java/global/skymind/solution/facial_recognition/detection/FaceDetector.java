package global.skymind.solution.facial_recognition.detection;

import org.bytedeco.opencv.opencv_core.Mat;

import java.util.List;

public class FaceDetector implements IFaceDetector {
    public static final String OPENCV_HAAR_CASCADE_FACEDETECTOR = "OPENCV_HAAR_CASCADE_FACEDETECTOR";
    public static final String OPENCV_DL_FACEDETECTOR = "OPENCV_DL_FACEDETECTOR";
    private int image_width;
    private int image_height;
    private double detection_threshold;

    public FaceDetector() {
    }

    @Override
    public void setImageWidth(int width) {
        this.image_width = width;
    }

    @Override
    public int getImage_width() {
        return this.image_width;
    }

    @Override
    public void setImageHeight(int height) {
        this.image_height = height;
    }

    @Override
    public int getImage_height() {
        return this.image_height;
    }

    @Override
    public void setDetectionThreshold(double detection_threshold) {
        this.detection_threshold = detection_threshold;
    }

    @Override
    public double getDetection_threshold() {
        return this.detection_threshold;
    }

    @Override
    public void detectFaces(Mat image) {

    }

    @Override
    public List<FaceLocalization> getFaceLocalization() {
        return null;
    }


}
