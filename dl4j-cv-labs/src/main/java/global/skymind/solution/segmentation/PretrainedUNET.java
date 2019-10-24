package global.skymind.solution.segmentation;

import global.skymind.solution.segmentation.imageUtils.visualisation;
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
import org.datavec.image.transform.*;
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
import net.lingala.zip4j.core.ZipFile;
import net.lingala.zip4j.exception.ZipException;

import javax.swing.*;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URL;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.bytedeco.opencv.global.opencv_imgproc.CV_RGB2GRAY;

public class PretrainedUNET {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(
            PretrainedUNET.class);

    public static final String featurizeExtractionLayer = "conv2d_5";
    protected static final long seed = 12345;
    protected static final int nEpochs = 1;
    private static final int height = 224;
    private static final int width = 224;
    private static final int channels = 1;
    private static final int batchSize = 4;
    private static final Random random = new Random(seed);

    public static void main(String[] args) throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException{

        ClassLoader classLoader = PretrainedUNET.class.getClassLoader();
        URL resource = classLoader.getResource("org/apache/http/message/BasicLineFormatter.class");
        System.out.println(resource);

        downloadData();
        unzipAllDataSet();

        ZooModel zooModel = UNet.builder().build();

        ComputationGraph unet = (ComputationGraph) zooModel.initPretrained(PretrainedType.SEGMENT);
        System.out.println(unet.summary());

        // Set listeners
        StatsStorage statsStorage = new InMemoryStatsStorage();
        StatsListener statsListener = new StatsListener(statsStorage);
        ScoreIterationListener scoreIterationListener= new ScoreIterationListener(1);

        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .updater(new Adam(new StepSchedule(ScheduleType.EPOCH,3e-4,0.5,5 )))
                .seed(seed)
                .build();

        //Construct a new model with the intended architecture and print summary
        ComputationGraph unetTransfer = new TransferLearning.GraphBuilder(unet)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor(featurizeExtractionLayer)
                .removeVertexAndConnections("activation_23")
                .nInReplace("conv2d_1",1, WeightInit.XAVIER)
                .nOutReplace("conv2d_23",1, WeightInit.XAVIER)
                .addLayer("output",
                        new CnnLossLayer.Builder(LossFunctions.LossFunction.XENT)
                                .activation(Activation.SIGMOID).build(),
                        "conv2d_23")
                .setOutputs("output")
                .build();

        System.out.println(unetTransfer.summary());

        unetTransfer.setListeners(statsListener, scoreIterationListener);

        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();
        uiServer.attach(statsStorage);

        File imagesPath = new File(System.getProperty("user.home"), ".deeplearning4j/data/data-science-bowl-2018/data-science-bowl-2018/data-science-bowl-2018-2/train/inputs");
        FileSplit imageFileSplit = new FileSplit(imagesPath, NativeImageLoader.ALLOWED_FORMATS, random);

        //Load labels
        CustomLabelGenerator labelMaker = new CustomLabelGenerator(height, width, 1); // labels have 1 channel

        BalancedPathFilter imageSplitPathFilter = new BalancedPathFilter(random, NativeImageLoader.ALLOWED_FORMATS, labelMaker);
        InputSplit[] imagesSplits = imageFileSplit.sample(imageSplitPathFilter, 0.8, 0.3);

        // Record reader
        ImageRecordReader imageRecordReaderTrain = new ImageRecordReader(height, width, channels, labelMaker);
        ImageRecordReader imageRecordReaderVal = new ImageRecordReader(height, width, channels, labelMaker);
        imageRecordReaderTrain.initialize(imagesSplits[0], getImageTransform());

        // Dataset iterator
        RecordReaderDataSetIterator imageDataSetTrain = new RecordReaderDataSetIterator(imageRecordReaderTrain, batchSize, 1, 1, true);


        // Preprocessing - normalisation
        DataNormalization dataNormalization = new ImagePreProcessingScaler(0,1);
        dataNormalization.fit(imageDataSetTrain);
        imageDataSetTrain.setPreProcessor(dataNormalization);

        // Visualisation -  training
        JFrame frame = visualisation.initFrame("Viz");
        JPanel panel = visualisation.initPanel(
                frame,
                1,
                height,
                width,
                1
        );

//        TRAINING
        for(int i=0; i<nEpochs; i++){

            log.info("Epoch: " + i);

            while(imageDataSetTrain.hasNext())
            {
                DataSet imageSet = imageDataSetTrain.next();

                unetTransfer.fit(imageSet);

                INDArray predict = unetTransfer.output(imageSet.getFeatures())[0];


                for (int n=0; n<imageSet.asList().size(); n++){
                    visualisation.visualize(
                            imageSet.get(n).getFeatures(),
                            imageSet.get(n).getLabels(),
//                            predict,
                            predict.get(NDArrayIndex.point(n)),
                            frame,
                            panel,
                            4,
                            224,
                            224
                    );
                }

            }

            imageDataSetTrain.reset();
        }

        // VALIDATION
        imageRecordReaderVal.initialize(imagesSplits[1]);
        RecordReaderDataSetIterator imageDataSetVal = new RecordReaderDataSetIterator(imageRecordReaderVal, batchSize, 1, 1, true);
        imageDataSetVal.setPreProcessor(dataNormalization);

        Evaluation eval = new Evaluation(2);

//         Visualisation -  validation
        JFrame frameVal = visualisation.initFrame("Viz");
        JPanel panelVal = visualisation.initPanel(
                frame,
                1,
                height,
                width,
                1
        );

        //            EXPORT IMAGES
        File exportDir = new File("export");

        if (!exportDir.exists() ) {
            exportDir.mkdir();
        }

        int count = 0;
        while(imageDataSetVal.hasNext()) {
            DataSet imageSetVal = imageDataSetVal.next();

            INDArray predict = unetTransfer.output(imageSetVal.getFeatures())[0];
            INDArray labels = imageSetVal.getLabels();

            if (count%5==0) {
                visualisation.export(exportDir, imageSetVal.getFeatures(), imageSetVal.getLabels(), predict, count );
            }

            count++;

            eval.eval(labels, predict);

            log.info(eval.stats());

//            Intersection over Union:  TP / (TP + FN + FP)
            float IOUNuclei = (float)eval.truePositives().get(1) / ((float)eval.truePositives().get(1) + (float)eval.falsePositives().get(1) + (float)eval.falseNegatives().get(1));

            System.out.println("IOU Cell Nuclei " + String.format("%.3f", IOUNuclei) );

            eval.reset();

            for (int n=0; n<imageSetVal.asList().size(); n++){
                visualisation.visualize(
                        imageSetVal.get(n).getFeatures(),
                        imageSetVal.get(n).getLabels(),
//                            predict,
                        predict.get(NDArrayIndex.point(n)),
                        frame,
                        panel,
                        4,
                        224,
                        224
                );
            }


        }

        // WRITE MODEL TO DISK
        File locationToSaveFineTune = new File(System.getProperty("user.home"),".deeplearning4j\\generated-models\\segmentUNetFineTune.zip");
        if (!locationToSaveFineTune.exists()){
            locationToSaveFineTune.getParentFile().mkdirs();
        }

        boolean saveUpdater = false;
        ModelSerializer.writeModel(unetTransfer, locationToSaveFineTune, saveUpdater);
        log.info("Model saved");
    }

    public static ImageTransform getImageTransform() {

//        ImageTransform noise = new NoiseTransform(random, (int) (height * width * 0.1));
//        ImageTransform enhanceContrast = new EqualizeHistTransform();
//        ImageTransform flip = new FlipImageTransform();
        ImageTransform rgb2gray = new ColorConversionTransform(CV_RGB2GRAY);
//        ImageTransform rotate = new RotateImageTransform(random, 30);

        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(
                new Pair<>(rgb2gray, 1.0)
//                new Pair<>(enhanceContrast, 1.0),
//                new Pair<>(flip, 0.5)
//                new Pair<>(rotate,0.5)
        );
        return new PipelineImageTransform(pipeline, false);
    }

    public static void downloadData() {
        // Download data
        File parentDir = new File(System.getProperty("user.home"), ".deeplearning4j\\data\\data-science-bowl-2018");
        String DATA_URL = "https://drive.google.com/a/skymind.my/uc?authuser=0&id=1zHn593J13dxLO1AJ0N2jKhpahs0yYGa0&export=download";

        File file = new File(parentDir + "\\data-science-bowl-2018.zip");

        if (!file.exists()) {
            file.getParentFile().mkdirs();
            HttpClientBuilder builder = HttpClientBuilder.create();
            CloseableHttpClient client = builder.build();
            try (CloseableHttpResponse response = client.execute(new HttpGet(DATA_URL))) {
                HttpEntity entity = response.getEntity();

                System.out.println(entity);

                if (entity != null) {
                    try (FileOutputStream outstream = new FileOutputStream(file)) {
                        entity.writeTo(outstream);
                        outstream.flush();
                    }
                }
            } catch (IOException ex) {
                System.out.println(ex);
            }


        }

    }

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
        File resourceDir = new File(System.getProperty("user.home"), ".deeplearning4j/data/data-science-bowl-2018");

        String zipClass0FilePath = resourceDir + "/data-science-bowl-2018.zip";

        File class0Folder = new File(resourceDir + "/data-science-bowl-2018");
        if (!class0Folder.exists()){
            System.out.println("Unzipping data ...");
            unzip(zipClass0FilePath, class0Folder.toString());
        }
    }
}