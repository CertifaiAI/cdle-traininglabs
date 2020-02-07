package global.skymind.solution.object_detection.AvocadoBananaDetector;

import global.skymind.Helper;
import org.apache.commons.io.FileUtils;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.util.ArchiveUtils;
import org.slf4j.Logger;
import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Paths;
import java.util.Random;

public class FruitDataSetIterator {

    private static final Logger log = org.slf4j.LoggerFactory.getLogger(FruitDataSetIterator.class);
    private static final int seed = 123;
    private static Random rng = new Random(seed);
    private static String dataDir;
    private static String downloadLink;
    private static String trainDir, testDir;
    private static FileSplit trainData, testData;
    private static final int nChannels = 3;
    public static final int gridWidth = 13;
    public static final int gridHeight = 13;
    public static final int yolowidth = 416;
    public static final int yoloheight = 416;

    private static RecordReaderDataSetIterator makeIterator(InputSplit split, String dir, int batchSize) throws Exception {

        ObjectDetectionRecordReader recordReader = new ObjectDetectionRecordReader(yoloheight, yolowidth, nChannels,
            gridHeight, gridWidth, new VocLabelProvider(dir));
        recordReader.initialize(split);
        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, 1, true);
        iter.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        return iter;
    }

    public static RecordReaderDataSetIterator trainIterator(int batchSize) throws Exception {
        return makeIterator(trainData, trainDir, batchSize);
    }

    public static RecordReaderDataSetIterator testIterator(int batchSize) throws Exception {
        return makeIterator(testData, testDir, batchSize);
    }

    public static void setup() throws IOException {
        log.info("Load data...");
        loadData();
        trainDir = dataDir.concat("/fruits/train/");
        testDir = dataDir.concat("/fruits/test/");
        trainData = new FileSplit(new File(trainDir), NativeImageLoader.ALLOWED_FORMATS, rng);
        testData = new FileSplit(new File(testDir), NativeImageLoader.ALLOWED_FORMATS, rng);
    }

    private static void loadData() throws IOException {
        dataDir = Paths.get(
                System.getProperty("user.home"),
                Helper.getPropValues("dl4j_home.data")
        ).toString();
        downloadLink = Helper.getPropValues("dataset.fruits.url");
        File parentDir = new File(Paths.get(dataDir,"fruits").toString());
        if(!parentDir.exists()){
            downloadAndUnzip();
        }
    }

    private static void downloadAndUnzip() throws IOException {
        String dataPath = new File(dataDir).getAbsolutePath();
        File zipFile = new File (dataPath,"fruits-detection.zip");

        if(!zipFile.isFile()){
            log.info("Downloading the dataset from "+downloadLink+ "...");
            FileUtils.copyURLToFile(new URL(downloadLink), zipFile);
        }
        ArchiveUtils.unzipFileTo(zipFile.getAbsolutePath(), dataPath);
    }
}
