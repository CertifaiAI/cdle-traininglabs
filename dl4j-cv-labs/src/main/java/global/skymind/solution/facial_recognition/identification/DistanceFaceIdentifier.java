package global.skymind.solution.facial_recognition.identification;

import global.skymind.solution.facial_recognition.detection.FaceLocalization;
import global.skymind.solution.facial_recognition.identification.feature.FaceFeatureProvider;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DistanceFaceIdentifier extends FaceIdentifier {
    private static final Logger log = LoggerFactory.getLogger(DistanceFaceIdentifier.class);
    private final FaceFeatureProvider _FaceFeatureProvider;
    private final int numPredictions;
    private final double threshold;
    private final int numSamples;

    public DistanceFaceIdentifier(FaceFeatureProvider faceFeatureProvider, File dictDir, int numPredictions, double threshold, int numSamples) throws IOException, ClassNotFoundException {
        this._FaceFeatureProvider = faceFeatureProvider;
        _FaceFeatureProvider.setupAnchor(dictDir);
        this.numPredictions = numPredictions;
        this.threshold = threshold;
        this.numSamples = numSamples;
    }

    @Override
    public List<List<Prediction>> recognize(List<FaceLocalization> faceLocalizations, Mat image) throws IOException {
        List<List<Prediction>> collection = new ArrayList<>();
        for (int i = 0; i<faceLocalizations.size(); i++) {

            int X = (int) faceLocalizations.get(i).getLeft_x();
            int Y = (int) faceLocalizations.get(i).getLeft_y();
            int Width = faceLocalizations.get(i).getValidWidth(image.size().width());
            int Height = faceLocalizations.get(i).getValidHeight(image.size().height());

            // Crop face, Resize and convert into INDArray
            Mat crop_image = new Mat(image, new Rect(X, Y, Width, Height));

            // Get a collection of predictions
            List<Prediction> predictions = _FaceFeatureProvider.predict(
                    crop_image, faceLocalizations.get(i), this.numPredictions, this.threshold, this.numSamples);
            collection.add(predictions);
        }
        return collection;
    }
}
