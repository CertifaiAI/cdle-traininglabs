package global.skymind.training.segmentation;

import global.skymind.Helper;
import global.skymind.solution.segmentation.CellDataSetIterator;
import global.skymind.training.segmentation.imageUtils.visualisation;
import net.lingala.zip4j.core.ZipFile;
import net.lingala.zip4j.exception.ZipException;
import org.apache.http.HttpEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClientBuilder;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ColorConversionTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.CnnLossLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.UNet;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;
import org.slf4j.Logger;

import javax.swing.*;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.bytedeco.opencv.global.opencv_imgproc.CV_RGB2GRAY;

public class PretrainedUNET {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(
            PretrainedUNET.class);

    private static final String featurizeExtractionLayer = "conv2d_4";
    private static final long seed = 12345;
    private static final int nEpochs = 30;
    private static final int height = 224;
    private static final int width = 224;
    private static final int batchSize = 4;
    private static final double trainPerc = 0.8;
    private static final Random random = new Random(seed);
    private static String modelExportDir;

    public static void main(String[] args) throws IOException {

        /*
         * Instructions for this lab exercise:
         * STEP 1: Import pretrained UNET (provided in model zoo).
         * STEP 2: Configuration of transfer learning.
         * STEP 3: Load data into RecordReaderDataSetIterator.
         * STEP 4: Run training.
         * STEP 5: We will use IOU (Intersection Over Union) as our evaluation metric. Complete the code for IOU calculation.
         *
         * */

//        //STEP 1: Import pretrained UNET (provided in model zoo)
//        ZooModel zooModel = UNet.builder().build();
//        ComputationGraph unet = (ComputationGraph) zooModel.initPretrained(PretrainedType.SEGMENT);
//        System.out.println(unet.summary());

        // Set listeners
        StatsStorage statsStorage = new InMemoryStatsStorage();
        StatsListener statsListener = new StatsListener(statsStorage);
        ScoreIterationListener scoreIterationListener= new ScoreIterationListener(1);

        //STEP 2: Configuration of transfer learning
        //STEP 2.1: Set updater and learning rate)
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
//                .updater() // 3.1 set updater here
                .seed(seed)
                .build();

//        //Construct a new model with the intended architecture and print summary
//        //STEP 2.2: Set which pre-trained layer to freeze and use as feature extractor
//        //STEP 2.3: Add a CnnLossLayer to form a Fully Convolutional Network
//        ComputationGraph unetTransfer = new TransferLearning.GraphBuilder(unet)
//                .fineTuneConfiguration(fineTuneConf)
//                .setFeatureExtractor(featurizeExtractionLayer)
//                    /* add CnnLossLayer (convolutional layer) to the pre-trained UNet to
//                    form a Fully Convolutional Network*/
//                .build();
//
//        System.out.println(unetTransfer.summary());
//
//        unetTransfer.setListeners(statsListener, scoreIterationListener);


//        //Initialize the user interface backend
//        UIServer uiServer = UIServer.getInstance();
//        uiServer.attach(statsStorage);
//
//        // STEP 3: Load data into RecordReaderDataSetIterator
//        CellDataSetIterator.setup(batchSize, trainPerc, getImageTransform());
//
//        //Create iterators
//        RecordReaderDataSetIterator imageDataSetTrain = CellDataSetIterator.trainIterator();
//        RecordReaderDataSetIterator imageDataSetVal = CellDataSetIterator.valIterator();

        // Visualisation -  training
        JFrame frame = visualisation.initFrame("Viz");
        JPanel panel = visualisation.initPanel(
                frame,
                1,
                height,
                width,
                1
        );

//        //STEP 4: Run training
//        for(int i=0; i<nEpochs; i++){
//
//            log.info("Epoch: " + i);
//
//            while(imageDataSetTrain.hasNext())
//            {
//                DataSet imageSet = imageDataSetTrain.next();
//
//                unetTransfer.fit(imageSet);
//
//                INDArray predict = unetTransfer.output(imageSet.getFeatures())[0];
//
//
//                for (int n=0; n<imageSet.asList().size(); n++){
//                    visualisation.visualize(
//                            imageSet.get(n).getFeatures(),
//                            imageSet.get(n).getLabels(),
//                            predict.get(NDArrayIndex.point(n)),
//                            frame,
//                            panel,
//                            4,
//                            224,
//                            224
//                    );
//                }
//
//            }
//
//            imageDataSetTrain.reset();
//        }

//        // VALIDATION
//        imageRecordReaderVal.initialize(imagesSplits[1]);
//        RecordReaderDataSetIterator imageDataSetVal = new RecordReaderDataSetIterator(imageRecordReaderVal, batchSize, 1, 1, true);
//        imageDataSetVal.setPreProcessor(dataNormalization);
//
//        Evaluation eval = new Evaluation(2);

        // VISUALISATION -  validation
        JFrame frameVal = visualisation.initFrame("Viz");
        JPanel panelVal = visualisation.initPanel(
                frame,
                1,
                height,
                width,
                1
        );

        // EXPORT IMAGES
        File exportDir = new File("export");

        if (!exportDir.exists() ) {
            exportDir.mkdir();
        }

//        float IOUtotal = 0;
//        int count = 0;
//        while(imageDataSetVal.hasNext()) {
//            DataSet imageSetVal = imageDataSetVal.next();
//
//            INDArray predict = unetTransfer.output(imageSetVal.getFeatures())[0];
//            INDArray labels = imageSetVal.getLabels();
//
//            if (count%5==0) {
//                visualisation.export(exportDir, imageSetVal.getFeatures(), imageSetVal.getLabels(), predict, count );
//            }
//
//            count++;
//
//            eval.eval(labels, predict);
//
//            log.info(eval.stats());
//
////            //STEP 5: Complete the code for IOU calculation here
////            float IOUNuclei = (float)eval.truePositives().get(1) / ((float)eval.truePositives().get(1) + (float)eval.falsePositives().get(1) + (float)eval.falseNegatives().get(1));
////            IOUtotal = IOUtotal + IOUNuclei;
////
////            System.out.println("IOU Cell Nuclei " + String.format("%.3f", IOUNuclei) );
//
//            eval.reset();
//
//            for (int n=0; n<imageSetVal.asList().size(); n++){
//                visualisation.visualize(
//                        imageSetVal.get(n).getFeatures(),
//                        imageSetVal.get(n).getLabels(),
//                        predict.get(NDArrayIndex.point(n)),
//                        frame,
//                        panel,
//                        4,
//                        224,
//                        224
//                );
//            }
//        }
//
//        System.out.print("Mean IOU: " + IOUtotal/count);

        // WRITE MODEL TO DISK
        // WRITE MODEL TO DISK
        modelExportDir = Paths.get(
                System.getProperty("user.home"),
                Helper.getPropValues("dl4j_home.generated-models")
        ).toString();


        File locationToSaveModel = new File(Paths.get(modelExportDir).toString() + "/segmentUNET.zip");
        if (!locationToSaveModel.exists()){
            locationToSaveModel.getParentFile().mkdirs();
        }

        boolean saveUpdater = false;
//        ModelSerializer.writeModel(unetTransfer, locationToSaveModel, saveUpdater);
        log.info("Model saved");
    }

    public static ImageTransform getImageTransform() {
        ImageTransform rgb2gray = new ColorConversionTransform(CV_RGB2GRAY);

        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(
                new Pair<>(rgb2gray, 1.0)
        );
        return new PipelineImageTransform(pipeline, false);
    }

}