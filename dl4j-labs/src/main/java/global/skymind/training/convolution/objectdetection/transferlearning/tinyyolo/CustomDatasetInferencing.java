package global.skymind.training.convolution.objectdetection.transferlearning.tinyyolo;

import global.skymind.training.convolution.objectdetection.transferlearning.tinyyolo.dataHelpers.NonMaxSuppression;
import global.skymind.training.convolution.objectdetection.transferlearning.tinyyolo.dataHelpers.XmlLabelProvider;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacpp.opencv_videoio;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
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
public class CustomDatasetInferencing {
    private static final Logger log = LoggerFactory.getLogger(CustomDatasetInferencing.class);
    private static final OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage();
    private static ComputationGraph model;

    // parameters matching the pretrained TinyYOLO model
    private static int width = 416;
    private static int height = 416;
    private static int nChannels = 3;
    private static int gridWidth = 13;
    private static int gridHeight = 13;
    private static Object[] labels;

    public static void main(String[] args) throws Exception {
        // number classes (digits) for the SVHN datasets
        int nClasses = 2;

        // parameters for the Yolo2OutputLayer
        int nBoxes = 5;
        double lambdaNoObj = 0.5;
        double lambdaCoord = 1.0;
        double[][] priorBoxes = {{2, 5}, {2.5, 6}, {3, 7}, {3.5, 8}, {4, 9}};
        double detectionThreshold = 0.08;

        // parameters for the training phase
        int batchSize = 1;
        int nEpochs = 20;
        double learningRate = 1e-4;
        double lrMomentum = 0.9;

        int seed = 123;
        Random rng = new Random(seed);

        File trainDir = new File("C:\\Users\\PK Chuah\\dl4jDataDir\\CustomDataset\\train_custom_objects");
        File testDir = new File("C:\\Users\\PK Chuah\\dl4jDataDir\\CustomDataset\\train_custom_objects");

        log.info("Load data...");

        FileSplit trainData = new FileSplit(trainDir, NativeImageLoader.ALLOWED_FORMATS, rng);
        FileSplit testData = new FileSplit(testDir, NativeImageLoader.ALLOWED_FORMATS, rng);

        ObjectDetectionRecordReader recordReaderTrain = new ObjectDetectionRecordReader(height, width, nChannels,
                        gridHeight, gridWidth, new XmlLabelProvider(trainDir));
        recordReaderTrain.initialize(trainData);
        ObjectDetectionRecordReader recordReaderTest = new ObjectDetectionRecordReader(height, width, nChannels,
                        gridHeight, gridWidth, new XmlLabelProvider(testDir));
        recordReaderTest.initialize(testData);

        // ObjectDetectionRecordReader performs regression, so we need to specify it here
        RecordReaderDataSetIterator train = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, 1, true);
        train.setPreProcessor(new ImagePreProcessingScaler(0, 1));
        RecordReaderDataSetIterator test = new RecordReaderDataSetIterator(recordReaderTest, 1, 1, 1, true);
        test.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        String modelFilename = "model.zip";

        if (new File(modelFilename).exists()) {
            log.info("Load model...");

            model = ModelSerializer.restoreComputationGraph(modelFilename);
        } else {
            log.info("Model not found.");
        }

        labels = train.getLabels().toArray();

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
        List<DetectedObject> objects = NonMaxSuppression.getObjects(objs);

        drawBoxes(image, objects);
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
