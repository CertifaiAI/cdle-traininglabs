//This example uses transfer learning from YOLOv2 pretrained model

package global.skymind.training.object_detection.MetalDefectsDetector;

import global.skymind.training.object_detection.dataHelpers.NonMaxSuppression;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.YOLO2;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.CV_8U;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.helper.opencv_core.RGB;

///**
// * This is an example of a metal surface defect detection using YOLOv2 architecture.
// * If no model exists, train a model using Transfer Learning, then validate with test set
// * If model exists, Validate model with test set.
// * Data Source: http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html
// *
// * This is a training copy, uncomment out the lines and try to complete it and execute the code.
// * **/

public class MetalSurfaceDefectDetector_YOLOv2 {
    private static final Logger log = LoggerFactory.getLogger(MetalSurfaceDefectDetector_YOLOv2.class);
    private static int seed = 123;
    private static double detectionThreshold = 0.5;
    private static int nBoxes = 5;
    private static double lambdaNoObj = 0.5;
    private static double lambdaCoord = 5.0;
    private static double[][] priorBoxes = {{1, 4}, {2.5, 6}, {3, 1}, {3.5, 8}, {4, 9}};

    private static int batchSize = 2;
    private static int nEpochs = 10;
    private static double learningRate = 1e-4;
    private static int nClasses = 6;
    private static List<String> labels;

    private static File modelFilename = new File(System.getProperty("user.dir"), "generated-models/MetalSurfaceDefects_yolov2.zip");
    private static ComputationGraph model;
    private static Frame frame = null;
    public static final Scalar BLUE = RGB(0, 0, 255);
    public static final Scalar GREEN = RGB(0, 255, 0);
    public static final Scalar RED = RGB(255, 0, 0);
    public static final Scalar YELLOW = RGB(255, 225, 0);
    public static final Scalar PINK = RGB(255, 0, 225);
    public static final Scalar CYAN = RGB(0, 225, 225);
    public static Scalar[] colormap = {BLUE, GREEN, RED, YELLOW, PINK, CYAN};
    private static String labeltext = null;

    public static void main(String[] args) throws Exception {

        MetalDefectDataSetIterator.setup();

        //        STEP 1 : Create iterators
//        RecordReaderDataSetIterator trainIter = null;
//        RecordReaderDataSetIterator testIter = null;
//
//        labels = null;

        //        If model does not exist, train the model, else directly go to model evaluation and then run real time object detection inference.
        if (modelFilename.exists()) {
            //        STEP 2 : Load trained model from previous execution
            Nd4j.getRandom().setSeed(seed);
            log.info("Load model...");
//            model = null;
        } else {
            Nd4j.getRandom().setSeed(seed);
            ComputationGraph pretrained = null;
            FineTuneConfiguration fineTuneConf = null;
            INDArray priors = Nd4j.create(priorBoxes);
            //     STEP 2 : Train the model using Transfer Learning
            //     STEP 2.1: Transfer Learning steps - Load TinyYOLO prebuilt model.
            log.info("Build model...");
            pretrained = (ComputationGraph) YOLO2.builder().build().initPretrained();

            //     STEP 2.2: Transfer Learning steps - Model Configurations.
//            fineTuneConf = null;

            //     STEP 2.3: Transfer Learning steps - Modify prebuilt model's architecture
//            model = getNewComputationGraph(pretrained, priors, fineTuneConf);

            System.out.println(model.summary(InputType.convolutional(
                    MetalDefectDataSetIterator.yoloheight,
                    MetalDefectDataSetIterator.yolowidth,
                    nClasses)));

            //     STEP 2.4: Training and Save model.
            log.info("Train model...");
            UIServer server = UIServer.getInstance();
            StatsStorage storage = new InMemoryStatsStorage();
            server.attach(storage);
            model.setListeners(new ScoreIterationListener(1), new StatsListener(storage));

//            for (int i = 1; i < nEpochs+1; i++) {
//                //write the training loop here
//            }

            ModelSerializer.writeModel(model, modelFilename, true);
            System.out.println("Model saved.");
        }
        //     STEP 3: Evaluate the model's accuracy by using the test iterator.
//        OfflineValidationWithTestDataset(testIter);

    }

    private static ComputationGraph getNewComputationGraph(ComputationGraph pretrained, INDArray priors, FineTuneConfiguration fineTuneConf) {

        return null;
    }

    private static FineTuneConfiguration getFineTuneConfiguration() {

        return null;
    }

    //    Evaluate visually the performance of the trained object detection model
    private static void OfflineValidationWithTestDataset(RecordReaderDataSetIterator test) throws InterruptedException {
        NativeImageLoader imageLoader = new NativeImageLoader();
        CanvasFrame canvas = new CanvasFrame("Validate Test Dataset");
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) model.getOutputLayer(0);
        Mat convertedMat = new Mat();
        Mat convertedMat_big = new Mat();

        while (test.hasNext() && canvas.isVisible()) {

            org.nd4j.linalg.dataset.DataSet ds = test.next();
            INDArray features = ds.getFeatures();
            INDArray results = model.outputSingle(features);
            List<DetectedObject> objs = yout.getPredictedObjects(results, detectionThreshold);
            List<DetectedObject> objects = NonMaxSuppression.getObjects(objs);

            Mat mat = imageLoader.asMat(features);
            mat.convertTo(convertedMat, CV_8U, 255, 0);
            int w = mat.cols() * 2;
            int h = mat.rows() * 2;
            resize(convertedMat, convertedMat_big, new Size(w, h));

            for (DetectedObject obj : objects) {
                double[] xy1 = obj.getTopLeftXY();
                double[] xy2 = obj.getBottomRightXY();
                String label = labels.get(obj.getPredictedClass());
                int x1 = (int) Math.round(w * xy1[0] / MetalDefectDataSetIterator.gridWidth);
                int y1 = (int) Math.round(h * xy1[1] / MetalDefectDataSetIterator.gridHeight);
                int x2 = (int) Math.round(w * xy2[0] / MetalDefectDataSetIterator.gridWidth);
                int y2 = (int) Math.round(h * xy2[1] / MetalDefectDataSetIterator.gridHeight);
                //Draw bounding box
                rectangle(convertedMat_big, new Point(x1, y1), new Point(x2, y2), colormap[obj.getPredictedClass()], 2, 0, 0);
                //Display label text
                labeltext = label + " " + (Math.round(obj.getConfidence() * 100.0) / 100.0) * 100.0 + "%";
                int[] baseline = {0};
                Size textSize = getTextSize(labeltext, FONT_HERSHEY_DUPLEX, 1, 1, baseline);
                rectangle(convertedMat_big, new Point(x1 + 2, y2 - 2), new Point(x1 + 2 + textSize.get(0), y2 - 2 - textSize.get(1)), colormap[obj.getPredictedClass()], FILLED, 0, 0);
                putText(convertedMat_big, labeltext, new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, RGB(0, 0, 0));
            }
            canvas.showImage(converter.convert(convertedMat_big));
            canvas.waitKey();
        }
        canvas.dispose();
    }
}



