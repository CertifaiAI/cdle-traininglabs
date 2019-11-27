package global.skymind.solution.facial_recognition.detection;

import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_dnn.Net;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_dnn.blobFromImage;
import static org.bytedeco.opencv.global.opencv_dnn.readNetFromCaffe;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

public class OpenCV_DeepLearningFaceDetector extends FaceDetector {

    private Net model;
    private int margin_percent = 0;
    private Mat detectedFaces;
    private int inputImageHeight;
    private int inputImageWidth;

    public OpenCV_DeepLearningFaceDetector(int imageWidth, int imageHeight, double detectionThreshold) {
        this.setImageHeight(imageHeight);
        this.setImageWidth(imageWidth);
        this.setDetectionThreshold(detectionThreshold);
        setModel();
    }

    private void setModel() {


        String PROTO_FILE = null;
        String CAFFE_MODEL_FILE = null;
        try {
            PROTO_FILE = new ClassPathResource("fdmodel/OpenCVDeepLearningFaceDetector/deploy.prototxt").getFile().getAbsolutePath();
            CAFFE_MODEL_FILE = new ClassPathResource("fdmodel/OpenCVDeepLearningFaceDetector/res10_300x300_ssd_iter_140000.caffemodel").getFile().getAbsolutePath();
        } catch (IOException e) {
            e.printStackTrace();
        }
        this.model = readNetFromCaffe(PROTO_FILE, CAFFE_MODEL_FILE);
    }

    @Override
    public void detectFaces(Mat image) {
        inputImageHeight = image.size().height();
        inputImageWidth = image.size().width();

        // resize the image to match the input size of the model
        resize(image, image, new Size(this.getImage_width(), this.getImage_height()));

        // create a 4-dimensional blob from image with NCHW (Number of images in the batch -for face_train only-, Channel, Height, Width) dimensions order,
        // for more detailes read the official docs at https://docs.opencv.org/trunk/d6/d0f/group__dnn.html#gabd0e76da3c6ad15c08b01ef21ad55dd8
        Mat blob = blobFromImage(image, 1.0,
                new Size(this.getImage_width(), this.getImage_height()),
                new Scalar(104.0, 177.0, 123.0, 0), false, false, CV_32F);

        // set the input to network model
        model.setInput(blob);
        // feed forward the input to the netwrok to get the output matrix
        detectedFaces = model.forward();
    }

    @Override
    public List<FaceLocalization> getFaceLocalization() {
        // extract a 2d matrix for 4d output matrix with form of (number of detections x 7)
        Mat ne = new Mat(new Size(detectedFaces.size(3), detectedFaces.size(2)), CV_32F, detectedFaces.ptr(0, 0));
        // create indexer to access elements of the matric
        FloatIndexer srcIndexer = ne.createIndexer();
        List<FaceLocalization> faceLocalizations = new ArrayList();
        for (int i = 0; i < detectedFaces.size(3); i++) {//iterate to extract elements
            float confidence = srcIndexer.get(i, 2);
            float f1 = srcIndexer.get(i, 3);
            float f2 = srcIndexer.get(i, 4);
            float f3 = srcIndexer.get(i, 5);
            float f4 = srcIndexer.get(i, 6);
            if (confidence > this.getDetection_threshold()) {
                //top left point's x
                float tx = f1 * inputImageWidth;
                //top left point's y
                float ty = f2 * inputImageHeight;
                //bottom right point's x
                float bx = f3 * inputImageWidth;
                //bottom right point's y
                float by = f4 * inputImageHeight;

                // add margin
                int w = (int) ((bx-tx)*margin_percent/100);
                int h = (int) ((by-ty)*margin_percent/100);
                tx = tx-w;
                ty = ty-h;
                bx = bx+w;
                by = by+w;

                faceLocalizations.add(new FaceLocalization(tx, ty, bx, by));
            }
        }
        return faceLocalizations;
    }
}
