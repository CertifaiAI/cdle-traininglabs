package global.skymind.solution.facial_recognition.identification;

import global.skymind.solution.facial_recognition.detection.FaceLocalization;
import org.bytedeco.opencv.opencv_core.Mat;

import java.io.IOException;
import java.util.List;

interface IFaceIdentifier {
    List<List<Prediction>> recognize(List<FaceLocalization> faceLocalizations, Mat image) throws IOException;
}
