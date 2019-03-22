package global.skymind.training.convolution.objectdetection.transferlearning.vgg16;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacpp.opencv_videoio;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicReference;

import static org.bytedeco.javacpp.opencv_imgproc.putText;
import static org.bytedeco.javacpp.opencv_videoio.CV_CAP_PROP_FRAME_HEIGHT;
import static org.bytedeco.javacpp.opencv_videoio.CV_CAP_PROP_FRAME_WIDTH;

/**
 * Example transfer learning from a Tiny YOLO model pretrained on ImageNet and Pascal VOC
 * to perform face recognition, model inference with model built in CustomDatasetTransferLearning.
 */
public class RealtimeInferencing {
    private static final Logger log = LoggerFactory.getLogger(RealtimeInferencing.class);
    private static final OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage();
    private static ComputationGraph model;

    // parameters matching the pretrained TinyYOLO model
    private static int width = 416;
    private static int height = 416;
    private static int nChannels = 3;
    private static int gridWidth = 13;
    private static int gridHeight = 13;

    // labels - model's classes
    private static Object[] labels = new String[]{"Amber Heard", "Jason Momoa", "Patrick Wilson"};

    // trained model file - default model.zip
    private static String modelFilename = "model.zip";

    // minimal confident of the detected faces to be display
    private static double detectionThreshold = 0.1;

    public static void main(String[] args) throws Exception {

        if (new File(modelFilename).exists()) {
            log.info("Load model...");
            model = ModelSerializer.restoreComputationGraph(modelFilename);
        } else {
            log.info("Model not found.");
        }

        // invoke webcam - inferencing
        final AtomicReference<opencv_videoio.VideoCapture> capture = new AtomicReference<>(new opencv_videoio.VideoCapture());
        capture.get().set(CV_CAP_PROP_FRAME_WIDTH, width);
        capture.get().set(CV_CAP_PROP_FRAME_HEIGHT, height);

        if (!capture.get().open(0)) {
            log.error("Can not open the cam !!!");
        }

        Mat colorimg = new Mat();
        CanvasFrame mainframe = new CanvasFrame("Real-time Detector", CanvasFrame.getDefaultGamma() / 2.2);
        mainframe.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
        mainframe.setCanvasSize(width, height);
        mainframe.setLocationRelativeTo(null);
        mainframe.setVisible(true);

        while (true) {
            while (capture.get().read(colorimg) && mainframe.isVisible()) {
                long st = System.currentTimeMillis();
                opencv_imgproc.resize(colorimg, colorimg, new opencv_core.Size(width, height));
                detect(colorimg, detectionThreshold);
                double per = (System.currentTimeMillis() - st) / 1000.0;
                log.info("It takes " + per + "Seconds to make detection");
                putText(colorimg, "Detection Time : " + per + " ms", new opencv_core.Point(10, 25), 2,.9, opencv_core.Scalar.YELLOW);

                mainframe.showImage(converter.convert(colorimg));
                try {
                    Thread.sleep(20);
                } catch (InterruptedException ex) {
                    log.error(ex.getMessage());
                }
            }
        }
    }

    public static void detect(Mat image, double threshold) {

        // access output layer
        org.deeplearning4j.nn.layers.objdetect.DetectedObject yout =
            (org.deeplearning4j.nn.layers.objdetect.DetectedObject) model.getOutputLayer(0);
        NativeImageLoader loader = new NativeImageLoader(width, height, nChannels);//, new ColorConversionTransform(COLOR_BGR2RGB)
        INDArray ds = null;
        try {
            ds = loader.asMatrix(image);
        } catch (IOException ex) {
            log.error(ex.getMessage());
        }
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(ds);
        INDArray results = model.outputSingle(ds);

        // get predicted classes (face) with confident higher than detection threshold
        int objs = yout.getPredictedClass();

        // draw boxes
        drawBoxes(image, objs);
    }

    private static void drawBoxes(Mat image, int objects) {

        System.out.println("Predicted class " + labels[objects]);
        putText(image, (String) labels[objects], new opencv_core.Point(0, 0), 1, .8, opencv_core.Scalar.RED);

    }
}
