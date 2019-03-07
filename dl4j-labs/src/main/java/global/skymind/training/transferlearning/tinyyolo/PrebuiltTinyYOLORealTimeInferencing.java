package global.skymind.training.transferlearning.tinyyolo;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacpp.opencv_videoio;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicReference;

import static org.bytedeco.javacpp.opencv_imgproc.putText;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import static org.bytedeco.javacpp.opencv_videoio.CV_CAP_PROP_FRAME_HEIGHT;
import static org.bytedeco.javacpp.opencv_videoio.CV_CAP_PROP_FRAME_WIDTH;


/**
 * Example transfer learning from a Tiny YOLO model pretrained on ImageNet and Pascal VOC
 * to perform object detection with bounding boxes on The Street View House Numbers (SVHN) Dataset.
 * <p>
 * References: <br>
 * - YOLO: Real-Time Object Detection: https://pjreddie.com/darknet/yolo/ <br>
 * - The Street View House Numbers (SVHN) Dataset: http://ufldl.stanford.edu/housenumbers/ <br>
 * <p>
 * Please note, cuDNN should be used to obtain reasonable performance: https://deeplearning4j.org/cudnn
 *
 * @author saudet
 */
public class PrebuiltTinyYOLORealTimeInferencing {
    private static final Logger log = LoggerFactory.getLogger(PrebuiltTinyYOLORealTimeInferencing.class);
    private static final OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage();
    private static ComputationGraph model;

    // parameters matching the pretrained TinyYOLO model
    private static int width = 416;
    private static int height = 416;
    private static int nChannels = 3;
    private static int gridWidth = 13;
    private static int gridHeight = 13;
    private static Object[] labels = new String[]{"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
        "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

    public static void main(String[] args) throws Exception {
        // number classes (digits) for the SVHN datasets
        int nClasses = 3;

        // parameters for the Yolo2OutputLayer
        int nBoxes = 5;
        double lambdaNoObj = 0.5;
        double lambdaCoord = 1.0;
        double[][] priorBoxes = {{2, 5}, {2.5, 6}, {3, 7}, {3.5, 8}, {4, 9}};
        double detectionThreshold = 0.6;

        // parameters for the training phase
        int batchSize = 1;
        int nEpochs = 20;
        double learningRate = 1e-4;
        double lrMomentum = 0.9;

        int seed = 123;
        Random rng = new Random(seed);

        ZooModel zooModel = TinyYOLO.builder().build();
        model = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);

        final AtomicReference<opencv_videoio.VideoCapture> capture = new AtomicReference<>(new opencv_videoio.VideoCapture());
        capture.get().set(CV_CAP_PROP_FRAME_WIDTH, width);
        capture.get().set(CV_CAP_PROP_FRAME_HEIGHT, height);

        if (!capture.get().open(0)) {
            log.error("Can not open the cam !!!");
        }

        Mat colorimg = new Mat();

        CanvasFrame mainframe = new CanvasFrame("Real-time Rubik's Cube Detector - Emaraic", CanvasFrame.getDefaultGamma() / 2.2);
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

    public static void detect(Mat image, double detectionthreshold) {
        org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout =
            (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) model.getOutputLayer(0);
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

        List<DetectedObject> objs = yout.getPredictedObjects(results, detectionthreshold);
//        List<DetectedObject> objects = NonMaxSuppression.getObjects(objs);
        drawBoxes(image, objs);//use objs to see the use of the NonMax Suppression algorithm
    }

    private static void drawBoxes(Mat image, List<DetectedObject> objects) {
        for (DetectedObject obj : objects) {
            double[] xy1 = obj.getTopLeftXY();
            double[] xy2 = obj.getBottomRightXY();
            int predictedClass = obj.getPredictedClass();
            System.out.println("Predicted class " + labels[predictedClass]);
            int x1 = (int) Math.round(width * xy1[0] / gridWidth);
            int y1 = (int) Math.round(height * xy1[1] / gridHeight);
            int x2 = (int) Math.round(width * xy2[0] / gridWidth);
            int y2 = (int) Math.round(height * xy2[1] / gridHeight);
            rectangle(image, new opencv_core.Point(x1, y1), new opencv_core.Point(x2, y2), opencv_core.Scalar.RED);
            putText(image, (String) labels[predictedClass], new opencv_core.Point(x1 + 2, y2 - 2), 1, .8, opencv_core.Scalar.RED);
        }
    }
}
