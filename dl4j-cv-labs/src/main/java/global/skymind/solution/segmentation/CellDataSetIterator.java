package global.skymind.solution.segmentation;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
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
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Random;

public class CellDataSetIterator {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(CellDataSetIterator.class);
    private static final int height = 224;
    private static final int width = 224;
    private static final int channels = 1;
    private static final int numClasses = 2;
    private static final long seed = 12345;
    private static final Random random = new Random(seed);
    private static File parentDir = new File(System.getProperty("user.home"), ".deeplearning4j\\data\\data-science-bowl-2018");
    private static File file = new File(parentDir + "\\data-science-bowl-2018.zip");
    private static String downloadLink = "https://drive.google.com/a/skymind.my/uc?authuser=0&id=1zHn593J13dxLO1AJ0N2jKhpahs0yYGa0&export=download";
    private static CustomLabelGenerator labelMaker = new CustomLabelGenerator(height, width, channels);
    private static InputSplit trainData,valData;
    private static int batchSize;

    //scale input to 0 - 1
    private static DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    private static ImageTransform transform;

    public CellDataSetIterator() throws IOException {
    }

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

        //Files in directories under the parent dir that have "allowed extensions" split needs a random number generator for reproducibility when splitting the files into train and test
        File imagesPath = new File(System.getProperty("user.home"), ".deeplearning4j/data/data-science-bowl-2018/data-science-bowl-2018/data-science-bowl-2018-2/train/inputs");
        FileSplit imageFileSplit = new FileSplit(imagesPath, NativeImageLoader.ALLOWED_FORMATS, random);

        BalancedPathFilter imageSplitPathFilter = new BalancedPathFilter(random, NativeImageLoader.ALLOWED_FORMATS, labelMaker);
        InputSplit[] imagesSplits = imageFileSplit.sample(imageSplitPathFilter, trainPerc, 1-trainPerc);

        trainData = imagesSplits[0];
        valData = imagesSplits[1];
    }

    public static void downloadData() {
        if (!file.exists()) {
            System.out.println("Creating dataset folder ...");
            file.getParentFile().mkdirs();
            HttpClientBuilder builder = HttpClientBuilder.create();
            CloseableHttpClient client = builder.build();
            System.out.println("Downloading dataset ...");
            try (CloseableHttpResponse response = client.execute(new HttpGet(downloadLink))) {
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
