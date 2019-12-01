package global.skymind.training.facial_recognition.identification;

import global.skymind.training.facial_recognition.detection.FaceLocalization;
import org.bytedeco.opencv.opencv_core.Mat;

import java.io.IOException;
import java.util.List;

public class FaceIdentifier implements IFaceIdentifier {
    public static final String FEATURE_DISTANCE_VGG16_PREBUILT = "FEATURE_DISTANCE_VGG16_PREBUILT";
    public static final String FEATURE_DISTANCE_RAMOK_FACENET_PREBUILT = "FEATURE_DISTANCE_RAMOK_FACENET_PREBUILT";

    public List<List<Prediction>> recognize(List<FaceLocalization> faceLocalizations, Mat image) throws IOException {
        return null;
    }
}
