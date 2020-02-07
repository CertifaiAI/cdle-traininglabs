package global.skymind.solution.segmentation.car;

import global.skymind.Helper;
import net.lingala.zip4j.core.ZipFile;
import net.lingala.zip4j.exception.ZipException;
import org.apache.http.HttpEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClientBuilder;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import global.skymind.solution.segmentation.imageUtils.CustomLabelGenerator;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class CarDataSetIterator {
    private static final int height = 224;
    private static final int width = 224;
    private static final int channels = 1;
    private static final long seed = 12345;
    private static final Random random = new Random(seed);
    private static String inputDir;
    private static String downloadLink;


    private static InputSplit trainData,valData;
    private static int batchSize;
    private static List<org.nd4j.linalg.primitives.Pair<String, String>> replacement = Arrays.asList(
            new org.nd4j.linalg.primitives.Pair<>("inputs","masks"),
            new org.nd4j.linalg.primitives.Pair<>(".jpg","_mask.gif")
    );
    private static CustomLabelGenerator labelMaker = new CustomLabelGenerator(height, width, channels,replacement);

    //scale input to 0 - 1
    private static DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    private static ImageTransform transform;

    public CarDataSetIterator() throws IOException {
    }

    //This method instantiates an ImageRecordReader and subsequently a RecordReaderDataSetIterator based on it
    private static RecordReaderDataSetIterator makeIterator(InputSplit split, boolean training) throws IOException {

        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
        if (training && transform != null){
            recordReader.initialize(split,transform);
        }else{
            recordReader.initialize(split);
        }
        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, 1, true);
        iter.setPreProcessor(scaler);

        return iter;
    }

    public static RecordReaderDataSetIterator trainIterator() throws IOException {
        return makeIterator(trainData, true);
    }

    public static RecordReaderDataSetIterator valIterator() throws IOException {
        return makeIterator(valData, false);
    }

    public static void setup(int batchSizeArg, double trainPerc, ImageTransform imageTransform) throws IOException {
        transform=imageTransform;
        setup(batchSizeArg,trainPerc);
    }

    public static void setup(int batchSizeArg, double trainPerc) throws IOException {

        downloadData();
        unzipAllDataSet();

        batchSize = batchSizeArg;

        inputDir = Paths.get(
                System.getProperty("user.home"),
                Helper.getPropValues("dl4j_home.data")
        ).toString();


        File imagesPath = new File(Paths.get(inputDir, "carvana-image-masking-challenge","train","inputs").toString());
        FileSplit imageFileSplit = new FileSplit(imagesPath, NativeImageLoader.ALLOWED_FORMATS, random);

        BalancedPathFilter imageSplitPathFilter = new BalancedPathFilter(random, NativeImageLoader.ALLOWED_FORMATS, labelMaker);
        InputSplit[] imagesSplits = imageFileSplit.sample(imageSplitPathFilter, trainPerc, 1-trainPerc);

        trainData = imagesSplits[0];
        valData = imagesSplits[1];
    }

    // Download dataset
    public static void downloadData() throws IOException {
        downloadLink = Helper.getPropValues("dataset.segmentationCar.url");

        inputDir =Paths.get(
                System.getProperty("user.home"),
                Helper.getPropValues("dl4j_home.data")
        ).toString();

        File dataZip = new File(Paths.get(inputDir,  "carvana-masking-challenge.zip").toString());

        if (!dataZip.exists()) {
            System.out.println("Creating dataset folder ...");
            dataZip.getParentFile().mkdir();
            HttpClientBuilder builder = HttpClientBuilder.create();
            CloseableHttpClient client = builder.build();
            System.out.println("Downloading dataset ...");
            try (CloseableHttpResponse response = client.execute(new HttpGet(downloadLink))) {
                HttpEntity entity = response.getEntity();

                System.out.println(entity);

                if (entity != null) {
                    try (FileOutputStream outstream = new FileOutputStream(dataZip)) {
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

    public static void unzipAllDataSet() throws IOException {
        //unzip training data set
        inputDir = Paths.get(
                System.getProperty("user.home"),
                Helper.getPropValues("dl4j_home.data")
        ).toString();

        File classFolder = new File(Paths.get(inputDir).toString());

        if (!classFolder.exists()){
            classFolder.mkdir();
            File zipClassFilePath = new File(Paths.get(inputDir, "carvana-masking-challenge.zip").toString());
            System.out.println("Unzipping dataset ...");
            unzip(zipClassFilePath.toString(), classFolder.toString());
        }
    }

}
