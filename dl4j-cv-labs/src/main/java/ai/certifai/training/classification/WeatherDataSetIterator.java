package ai.certifai.solution.classification;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class WeatherDataSetIterator {

    private static final Logger log = org.slf4j.LoggerFactory.getLogger(WeatherDataSetIterator.class);

    private static final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static Random rngseed = new Random(123);

    private static final int height = 224;
    private static final int width = 224;
    private static final int channels = 3;
    private static final int numOutput = 4;

    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private static InputSplit trainData, testData;
    private static int batchSize;

    public static DataSetIterator trainIterator() throws IOException {
        return makeIterator(trainData);
    }

    public static DataSetIterator testIterator() throws IOException {
        return makeIterator(testData);
    }

    public static void setup(int batchSizeArg, int trainPerc) throws IOException, IllegalAccessException {

        batchSize = batchSizeArg;
        File parentDir = new ClassPathResource("WeatherImage").getFile();
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, rngseed);
        BalancedPathFilter pathFilter = new BalancedPathFilter(rngseed, allowedExtensions, labelMaker);
        if (trainPerc >= 100) {
            throw new IllegalAccessException("Percentage of data split for training set has to be less than 100%");
        }
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, trainPerc, 100-trainPerc);
        trainData = filesInDirSplit[0];
        testData = filesInDirSplit[1];

    }

    private static DataSetIterator makeIterator (InputSplit split) throws IOException {
        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
        recordReader.initialize(split);
        DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numOutput);
        iter.setPreProcessor(new VGG16ImagePreProcessor());
        return iter;
    }

}
