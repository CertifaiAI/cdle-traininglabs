//This example uses transfer learning from TinyYOLO pretrained model

package global.skymind.solution.object_detection;

import global.skymind.solution.object_detection.dataHelpers.LabelImgXmlLabelProvider;
import global.skymind.solution.object_detection.dataHelpers.NonMaxSuppression;
import net.lingala.zip4j.core.ZipFile;
import net.lingala.zip4j.exception.ZipException;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.transform.ColorConversionTransform;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.event.KeyEvent;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Random;

import static org.bytedeco.opencv.global.opencv_core.CV_8U;
import static org.bytedeco.opencv.global.opencv_core.flip;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.helper.opencv_core.RGB;

public class RealTimeABDetector_tiny {
    private static final Logger log = LoggerFactory.getLogger(RealTimeABDetector_tiny.class);
    private static int nChannels = 3;
    private static final int gridWidth = 13;
    private static final int gridHeight = 13;
    private static double detectionThreshold = 0.3;
    private static final int tinyyolowidth = 416;
    private static final int tinyyoloheight = 416;

    private static int nBoxes = 5;
    private static double lambdaNoObj = 0.5;
    private static double lambdaCoord = 5.0;
    private static double[][] priorBoxes = {{1, 3}, {2.5, 6}, {3, 4}, {3.5, 8}, {4, 9}};


    private static int batchSize = 4;
    private static int nEpochs = 40;

    private static double learningRate = 1e-4;

    private static int nClasses = 2;
    private static List<String> labels;
    private static int seed = 123;
    private static Random rng = new Random(seed);
    private static File modelFilename = new File(System.getProperty("user.dir"),"generated-models/Avocado_Banana_Detector_tinyyolo.zip");
    private static ComputationGraph model;
    private static Frame frame = null;
    public static final Scalar GREEN = RGB(0, 255.0, 0);
    public static final Scalar YELLOW = RGB(255, 255, 0);
    public static Scalar[] colormap = {GREEN,YELLOW};
    public static String labeltext = null;

    public static void main(String[] args) throws Exception {

        //         STEP 1 : Unzip the dataset into your local pc.
        unzipAllDataSet();
        //         STEP 2 : Specify your training data and test data.
        File trainDir = new File(System.getProperty("user.home"), ".deeplearning4j/data/fruits/train/");
        File testDir = new File(System.getProperty("user.home"), ".deeplearning4j/data/fruits/test/");
        log.info("Load data...");
        FileSplit trainData = new FileSplit(trainDir, NativeImageLoader.ALLOWED_FORMATS, rng);
        FileSplit testData = new FileSplit(testDir, NativeImageLoader.ALLOWED_FORMATS, rng);

        //         STEP 3 : Load the data into a RecordReader and make it into a RecordReaderDatasetIterator. MinMax scaling was applied as a Preprocessing step.
        ObjectDetectionRecordReader recordReaderTrain = new ObjectDetectionRecordReader(tinyyoloheight, tinyyolowidth, nChannels,
                gridHeight, gridWidth, new LabelImgXmlLabelProvider(trainDir));

        recordReaderTrain.initialize(trainData);
        ObjectDetectionRecordReader recordReaderTest = new ObjectDetectionRecordReader(tinyyoloheight, tinyyolowidth, nChannels,
                gridHeight, gridWidth, new LabelImgXmlLabelProvider(testDir));

        recordReaderTest.initialize(testData);

        RecordReaderDataSetIterator train = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, 1, true);
        train.setPreProcessor(new ImagePreProcessingScaler(0, 1));
        RecordReaderDataSetIterator test = new RecordReaderDataSetIterator(recordReaderTest, 1, 1, 1, true);
        test.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        //         STEP 3.1 : Determine the list of available labels
        labels = train.getLabels();

        //        If model does not exist, train the model, else directly go to model evaluation and then run real time object detection inference.

        if (modelFilename.exists()) {
        //        STEP 4 : Load trained model from previous execution
            Nd4j.getRandom().setSeed(seed);
            log.info("Load model...");
            model = ModelSerializer.restoreComputationGraph(modelFilename);
        } else {
            Nd4j.getRandom().setSeed(seed);
            ComputationGraph pretrained = null;
            FineTuneConfiguration fineTuneConf = null;
            INDArray priors = Nd4j.create(priorBoxes);
            //     STEP 4 : Train the model using Transfer Learning

            //     STEP 4.1: Transfer Learning steps - Load TinyYOLO prebuilt model.
            log.info("Build model...");
            pretrained = (ComputationGraph) TinyYOLO.builder().build().initPretrained();

            //     STEP 4.2: Transfer Learning steps - Model Configurations.
            fineTuneConf = getFineTuneConfiguration();

            //     STEP 4.3: Transfer Learning steps - Modify prebuilt model's architecture
            model = getNewComputationGraph(pretrained, priors, fineTuneConf);
            System.out.println(model.summary(InputType.convolutional(tinyyoloheight, tinyyolowidth, nClasses)));

            //     STEP 4.4: Training and Save model.
            log.info("Train model...");
            UIServer server = UIServer.getInstance();
            StatsStorage storage = new InMemoryStatsStorage();
            server.attach(storage);
            model.setListeners(new ScoreIterationListener(1), new StatsListener(storage));

            for (int i = 1; i < nEpochs+1; i++) {
                train.reset();
                while (train.hasNext()) {
                    model.fit(train.next());
                }
                log.info("*** Completed epoch {} ***", i);
            }
            ModelSerializer.writeModel(model, modelFilename, true);
            System.out.println("Model saved.");
        }
        //     STEP 5: Training and Save model.
        OfflineValidationWithTestDataset(test);
        //     STEP 6: Training and Save model.
        doInference();
    }

    private static ComputationGraph getNewComputationGraph(ComputationGraph pretrained, INDArray priors, FineTuneConfiguration fineTuneConf) {
        ComputationGraph _ComputationGraph = new TransferLearning.GraphBuilder(pretrained)
                .fineTuneConfiguration(fineTuneConf)
                .removeVertexKeepConnections("conv2d_9")
                .removeVertexKeepConnections("outputs")
                .addLayer("conv2d_9",
                        new ConvolutionLayer.Builder(1, 1)
                                .nIn(1024)
                                .nOut(nBoxes * (5 + nClasses))
                                .stride(1, 1)
                                .convolutionMode(ConvolutionMode.Same)
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.IDENTITY)
                                .build(),
                        "leaky_re_lu_8")
                .addLayer("outputs",
                        new Yolo2OutputLayer.Builder()
                                .lambbaNoObj(lambdaNoObj)
                                .lambdaCoord(lambdaCoord)
                                .boundingBoxPriors(priors.castTo(DataType.FLOAT))
                                .build(),
                        "conv2d_9")
                .setOutputs("outputs")
                .build();

        return _ComputationGraph;
    }

    private static FineTuneConfiguration getFineTuneConfiguration() {

        FineTuneConfiguration _FineTuneConfiguration = new FineTuneConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(1.0)
                .updater(new Adam.Builder().learningRate(learningRate).build())
                .l2(0.00001)
                .activation(Activation.IDENTITY)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .build();

        return _FineTuneConfiguration;
    }

//    Evaluate visually the performance of the trained object detection model
    private static void OfflineValidationWithTestDataset(RecordReaderDataSetIterator test)throws InterruptedException{
        NativeImageLoader imageLoader = new NativeImageLoader();
        CanvasFrame canvas = new CanvasFrame("Validate Test Dataset");
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer)model.getOutputLayer(0);
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
            resize(convertedMat,convertedMat_big, new Size(w, h));

            for (DetectedObject obj : objects) {
                double[] xy1 = obj.getTopLeftXY();
                double[] xy2 = obj.getBottomRightXY();
                String label = labels.get(obj.getPredictedClass());
                int x1 = (int) Math.round(w * xy1[0] / gridWidth);
                int y1 = (int) Math.round(h * xy1[1] / gridHeight);
                int x2 = (int) Math.round(w * xy2[0] / gridWidth);
                int y2 = (int) Math.round(h * xy2[1] / gridHeight);
                //Draw bounding box
                rectangle(convertedMat_big, new Point(x1, y1), new Point(x2, y2), colormap[obj.getPredictedClass()], 2, 0, 0);
                //Display label text
                labeltext =label+" "+(Math.round(obj.getConfidence()*100.0)/100.0)*100.0 +"%";
                int baseline[]={0};
                Size textSize=getTextSize(labeltext, FONT_HERSHEY_DUPLEX, 1,1,baseline);
                rectangle(convertedMat_big, new Point(x1 + 2, y2 - 2), new Point(x1 + 2+textSize.get(0), y2 - 2-textSize.get(1)), colormap[obj.getPredictedClass()], FILLED,0,0);
                putText(convertedMat_big, labeltext, new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, RGB(0,0,0));
            }
            canvas.showImage(converter.convert(convertedMat_big));
            canvas.waitKey();
        }
        canvas.dispose();
    }

    // Stream video frames from Webcam and run them through TinyYOLO model and get predictions
    private static void doInference(){

        String cameraPos = "front";
        int cameraNum = 0;
        Thread thread = null;
        NativeImageLoader loader = new NativeImageLoader(tinyyolowidth, tinyyoloheight, 3, new ColorConversionTransform(COLOR_BGR2RGB));
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);

        if( !cameraPos.equals("front") && !cameraPos.equals("back") )
        {
            try {
                throw new Exception("Unknown argument for camera position. Choose between front and back");
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        FrameGrabber grabber = null;
        try {
            grabber = FrameGrabber.createDefault(cameraNum);
        } catch (FrameGrabber.Exception e) {
            e.printStackTrace();
        }
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

        try {
            grabber.start();
        } catch (FrameGrabber.Exception e) {
            e.printStackTrace();
        }

        String winName = "Object Detection";
        CanvasFrame canvas = new CanvasFrame(winName);

        int w = grabber.getImageWidth();
        int h = grabber.getImageHeight();


        canvas.setCanvasSize(w, h);
        while (true)
        {
            try {
                frame = grabber.grab();
            } catch (FrameGrabber.Exception e) {
                e.printStackTrace();
            }

            //if a thread is null, create new thread
            if (thread == null)
            {
                thread = new Thread(() ->
                {
                    while (frame != null)
                    {
                        try
                        {
                            Mat rawImage = new Mat();

                            //Flip the camera if opening front camera
                            if(cameraPos.equals("front"))
                            {
                                Mat inputImage = converter.convert(frame);
                                flip(inputImage, rawImage, 1);
                            }
                            else
                            {
                                rawImage = converter.convert(frame);
                            }

                            Mat resizeImage = new Mat();
                            resize(rawImage, resizeImage, new Size(tinyyolowidth, tinyyoloheight));

                            INDArray inputImage = loader.asMatrix(resizeImage);
                            scaler.transform(inputImage);
                            INDArray outputs = model.outputSingle(inputImage);
                            org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer)model.getOutputLayer(0);
                            List<DetectedObject> objs = yout.getPredictedObjects(outputs, detectionThreshold);
                            List<DetectedObject> objects = NonMaxSuppression.getObjects(objs);

                            for (DetectedObject obj : objects) {

                                double[] xy1 = obj.getTopLeftXY();
                                double[] xy2 = obj.getBottomRightXY();
                                String label = labels.get(obj.getPredictedClass());
                                int x1 = (int) Math.round(w * xy1[0] / gridWidth);
                                int y1 = (int) Math.round(h * xy1[1] / gridHeight);
                                int x2 = (int) Math.round(w * xy2[0] / gridWidth);
                                int y2 = (int) Math.round(h * xy2[1] / gridHeight);
                                //Draw bounding box
                                rectangle(rawImage, new Point(x1, y1), new Point(x2, y2), colormap[obj.getPredictedClass()], 2, 0, 0);
                                //Display label text
                                labeltext =label+" "+(Math.round(obj.getConfidence()*100.0)/100.0)*100.0 +"%";
                                int baseline[]={0};
                                Size textSize=getTextSize(labeltext, FONT_HERSHEY_DUPLEX, 1,1,baseline);
                                rectangle(rawImage, new Point(x1 + 2, y2 - 2), new Point(x1 + 2+textSize.get(0), y2 - 2-textSize.get(1)), colormap[obj.getPredictedClass()], FILLED,0,0);
                                putText(rawImage, labeltext, new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, RGB(0,0,0));
                            }
                            canvas.showImage(converter.convert(rawImage));
                        }
                        catch (Exception e)
                        {
                            throw new RuntimeException(e);
                        }
                    }
                });
                thread.start();
            }

            KeyEvent t = null;
            try {
                t = canvas.waitKey(33);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            if ((t != null) && (t.getKeyCode() == KeyEvent.VK_Q)) {
                break;
            }
        }
    }

    //To unzip the training and test datset
    public static void unzip(String source, String destination){
        try {
            ZipFile zipFile = new ZipFile(source);
            zipFile.extractAll(destination);
        } catch (ZipException e) {
            e.printStackTrace();
        }
    }

    public static void unzipAllDataSet(){
        //unzip training data set
        File resourceDir = new File(System.getProperty("user.home"), ".deeplearning4j/data/fruits");
        if (!resourceDir.exists()) resourceDir.mkdirs();

        String zipTrainFilePath = null;
        String zipTestFilePath = null;
        try {
            zipTrainFilePath  = new ClassPathResource("fruits/train.zip").getFile().toString();
            zipTestFilePath  = new ClassPathResource("fruits/test.zip").getFile().toString();
            System.out.println(zipTrainFilePath);
            System.out.println(zipTestFilePath);
        } catch (IOException e) {
            e.printStackTrace();
        }
        File trainFolder = new File(resourceDir+"/train");
        if (!trainFolder.exists()) unzip(zipTrainFilePath, resourceDir.toString());
        System.out.println("unziptrain done");
        System.out.println(trainFolder);

        File testFolder = new File(resourceDir+"/test");
        if (!testFolder.exists()) unzip(zipTestFilePath, resourceDir.toString());
        System.out.println(testFolder);
        System.out.println("unziptest done");
    }
}



