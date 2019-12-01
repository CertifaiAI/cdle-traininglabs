package global.skymind.solution.facial_recognition.identification.feature;

import global.skymind.solution.facial_recognition.detection.FaceLocalization;
import global.skymind.solution.facial_recognition.identification.Prediction;
import org.bytedeco.opencv.opencv_core.Mat;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

interface IFaceFeatureProvider {
    INDArray getEmbeddings(INDArray arr);
    ArrayList<LabelFeaturePair> setupAnchor(File classDict) throws IOException, ClassNotFoundException;
    List<Prediction> predict(Mat image, FaceLocalization faceLocalization, double threshold, int numSamples) throws IOException;

}
