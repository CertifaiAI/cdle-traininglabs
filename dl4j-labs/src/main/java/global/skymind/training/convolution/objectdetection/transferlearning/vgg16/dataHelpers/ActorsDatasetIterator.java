package global.skymind.training.convolution.objectdetection.transferlearning.vgg16.dataHelpers;

import org.apache.commons.io.FileUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.util.ArchiveUtils;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.Random;

public class ActorsDatasetIterator extends RecordReaderDataSetIterator {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(ActorsDatasetIterator.class);

    private static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static final Random rng  = new Random(13);

    private static final int height = 224;
    private static final int width = 224;
    private static final int channels = 3;
    private static final int numClasses = 3;

    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private static InputSplit trainData,testData;
    private static int batchSize;

    public ActorsDatasetIterator(RecordReader recordReader, int batchSize) {
        super(recordReader, batchSize);
    }

    public static RecordReaderDataSetIterator trainIterator() throws IOException {
        return makeIterator(trainData, batchSize);

    }

    public static RecordReaderDataSetIterator testIterator() throws IOException {
        return makeIterator(testData, 1);
    }

    public static void setup(File parentDir, int batchSizeArg, int trainPerc) throws IOException {

        batchSize = batchSizeArg;
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, rng);
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedExtensions, labelMaker);
        if (trainPerc >= 100) {
            throw new IllegalArgumentException("Percentage of data set aside for training has to be less than 100%. Test percentage = 100 - training percentage, has to be greater than 0");
        }
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, trainPerc, 100-trainPerc);
        trainData = filesInDirSplit[0];
        testData = filesInDirSplit[1];
    }

    private static RecordReaderDataSetIterator makeIterator(InputSplit split, int batchSize) throws IOException {
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        recordReader.initialize(split);
        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses);
        return iter;
    }
}
