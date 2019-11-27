package global.skymind.training.facial_recognition.detection;

import org.bytedeco.opencv.opencv_core.Mat;

import java.util.List;

interface IFaceDetector {
    void setImageWidth(int width);
    int getImage_width();
    void setImageHeight(int height);
    int getImage_height();
    void setDetectionThreshold(double threshold);
    double getDetection_threshold();
    void detectFaces(Mat image);
    List<FaceLocalization> getFaceLocalization();
}

