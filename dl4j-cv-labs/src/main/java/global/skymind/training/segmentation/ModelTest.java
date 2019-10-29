package global.skymind.training.segmentation;

import global.skymind.training.segmentation.imageUtils.visualisation;
import org.apache.http.HttpEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClientBuilder;
import org.bytedeco.opencv.opencv_core.Mat;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ColorConversionTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.bytedeco.opencv.global.opencv_core.CV_8UC1;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class ModelTest {
    private static final Logger log = LoggerFactory.getLogger(ModelTest.class);
    private static File modelFilename = new File(System.getProperty("user.home"), ".deeplearning4j/generated-models/segmentUNetFineTune.zip");
    private static final int height = 224;
    private static final int width = 224;
    private static final int channels = 1;
    protected static final long seed = 12345;
    private static final Random random = new Random(seed);
    private static ComputationGraph model;

    public static void main(String[] args) throws Exception {

        /*
        * This program will perform segmentation on the test dataset, based on either model trained by students or model provided by instructor
        *
        * */

        if (modelFilename.exists()) {
            log.info("Load model...");
            try {
                model = ModelSerializer.restoreComputationGraph(modelFilename);
            } catch(Exception ex){
                ex.printStackTrace();
            }
        } else {
            log.info("Downloading pre-trained model...");
            downloadModel();
            try {
                model = ModelSerializer.restoreComputationGraph(modelFilename);
            } catch(Exception ex){
                ex.printStackTrace();
            }

        }

        File testImagesPath = new File(System.getProperty("user.home"), ".deeplearning4j/data/data-science-bowl-2018/data-science-bowl-2018/data-science-bowl-2018-2/test/inputs");
        FileSplit imageSplit = new FileSplit(testImagesPath, NativeImageLoader.ALLOWED_FORMATS, random);

        // Instantiate label generator
        CustomLabelGenerator labelMaker = new CustomLabelGenerator(height, width, 1); // labels have 1 channel

        // Initialize recordreader
        ImageRecordReader imageRecordReaderTest = new ImageRecordReader(height, width, channels, labelMaker);
        imageRecordReaderTest.initialize(imageSplit, getImageTransform());

        // Dataset iterator
        RecordReaderDataSetIterator imageDataSetTest = new RecordReaderDataSetIterator(imageRecordReaderTest, 1, 1, 1, true);

        // Preprocessing - normalisation
        DataNormalization dataNormalization = new ImagePreProcessingScaler(0,1);
        dataNormalization.fit(imageDataSetTest);
        imageDataSetTest.setPreProcessor(dataNormalization);

        // Visualisation of test image prediction
        JFrame frame = visualisation.initFrame("Viz");
        JPanel panel = visualisation.initPanel(
                frame,
                1,
                height,
                width,
                1
        );

        // Inference and evaluation on test set
        Evaluation eval = new Evaluation(2);

        float iou = 0;
        int count = 0;

        NativeImageLoader loader = new NativeImageLoader();

        while(imageDataSetTest.hasNext())
        {
            DataSet imageSet = imageDataSetTest.next();

            INDArray predict = model.output(imageSet.getFeatures())[0];

            INDArray labels = imageSet.getLabels();

            eval.eval(labels, predict);
            log.info(eval.stats());

            //Intersection over Union:  TP / (TP + FN + FP)
            float IOUNuclei = (float)eval.truePositives().get(1) / ((float)eval.truePositives().get(1) + (float)eval.falsePositives().get(1) + (float)eval.falseNegatives().get(1));
            System.out.println("IOU Cell Nuclei " + String.format("%.3f", IOUNuclei) );

            iou = iou + IOUNuclei;
            count++;

            eval.reset();


            for (int n=0; n<imageSet.asList().size(); n++){
                visualisation.visualize(
                        imageSet.get(n).getFeatures(),
                        imageSet.get(n).getLabels(),
                        predict.get(NDArrayIndex.point(n)),
                        frame,
                        panel,
                        4,
                        224,
                        224
                );
            }
        }
        System.out.print("Summed Iou: " + iou);
        System.out.print("num samples: " + count);
        System.out.print("Mean Iou: "+ iou/count );
    }


    public static void downloadModel() {
        // Download trained model
        File parentDir = new File(System.getProperty("user.home"), ".deeplearning4j\\generated-models");
        String DATA_URL = "https://uc708c152a8b5bf05cc90d34c1dd.dl.dropboxusercontent.com/cd/0/get/ArU2hEyhanrptTOljm8JIHRiPmjnh7R9DjPxf1mADj8cEdFZCwy3wyWvFqwn2T_rxGkWXc_mTd64-2oTkru5QM4lKkGCJtoH3tb3eZBhEDZON8vsNWuF0lx3_rbKSPPV7n4/file";

        File file = new File(parentDir + "\\segmentUNetFineTune.zip");

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

    public static ImageTransform getImageTransform() {

        ImageTransform rgb2gray = new ColorConversionTransform(CV_RGB2GRAY);

        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(
                new Pair<>(rgb2gray, 1.0)
        );
        return new PipelineImageTransform(pipeline, false);
    }

    public static INDArray dilateImage(INDArray src, NativeImageLoader loader) throws IOException {
        Mat mat = loader.asMat(src);
        Mat kernel = Mat.ones(3, 3, CV_8UC1).asMat();
        dilate(mat, mat, kernel);

        return loader.asMatrix(mat);
    }

    public static INDArray erodeImage(INDArray src, NativeImageLoader loader) throws IOException {
        Mat mat = loader.asMat(src);
        Mat kernel = Mat.ones(3, 3, CV_8UC1).asMat();
        erode(mat, mat, kernel);

        return loader.asMatrix(mat);
    }


}