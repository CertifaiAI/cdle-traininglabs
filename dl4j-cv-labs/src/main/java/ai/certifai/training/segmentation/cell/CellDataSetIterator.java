package ai.certifai.training.segmentation.cell;

import ai.certifai.Helper;
import ai.certifai.training.utilities.DataUtilities;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Random;

public class CellDataSetIterator {
    private static final int height = 224;
    private static final int width = 224;
    private static final int channels = 1;
    private static final long seed = 12345;
    private static final Random random = new Random(seed);
    private static String inputDir;
    private static String downloadLink;
    private static CustomLabelGenerator labelMaker = new CustomLabelGenerator(height, width, channels);
    private static InputSplit trainData, valData;
    private static int batchSize;

    //scale input to 0 - 1
    private static DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    private static ImageTransform transform;

    public CellDataSetIterator() throws IOException {
    }

    //This method instantiates an ImageRecordReader and subsequently a RecordReaderDataSetIterator based on it
    private static RecordReaderDataSetIterator makeIterator(InputSplit split, boolean training) throws IOException {
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        if (training && transform != null) {
            recordReader.initialize(split, transform);
        } else {
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
        transform = imageTransform;
        setup(batchSizeArg, trainPerc);
    }

    //This method does the following:
    // 1. Download and unzip dataset if it hasn't been downloaded
    // 2. Split dataset into training set and validation set
    public static void setup(int batchSizeArg, double trainPerc) throws IOException {

        downloadLink = Helper.getPropValues("dataset.segmentationCell.url");
        inputDir = Paths.get(
                System.getProperty("user.home"),
                Helper.getPropValues("dl4j_home.data")
        ).toString();

        File dataZip = new File(Paths.get(inputDir, "data-science-bowl-2018", "data-science-bowl-2018.zip").toString());
        File classFolder = new File(Paths.get(inputDir, "data-science-bowl-2018", "data-science-bowl-2018").toString());
        if (!dataZip.exists()) {
            if (DataUtilities.downloadFile(downloadLink, dataZip.getAbsolutePath())) {
                DataUtilities.extractZip(dataZip.getAbsolutePath(), classFolder.getAbsolutePath());
            }
        }

        batchSize = batchSizeArg;

        inputDir = Paths.get(
                System.getProperty("user.home"),
                Helper.getPropValues("dl4j_home.data")
        ).toString();

        File imagesPath = new File(Paths.get(inputDir, "data-science-bowl-2018", "data-science-bowl-2018", "data-science-bowl-2018-2", "train", "inputs").toString());
        FileSplit imageFileSplit = new FileSplit(imagesPath, NativeImageLoader.ALLOWED_FORMATS, random);
        BalancedPathFilter imageSplitPathFilter = new BalancedPathFilter(random, NativeImageLoader.ALLOWED_FORMATS, labelMaker);
        InputSplit[] imagesSplits = imageFileSplit.sample(imageSplitPathFilter, trainPerc, 1 - trainPerc);

        trainData = imagesSplits[0];
        valData = imagesSplits[1];
    }
}
