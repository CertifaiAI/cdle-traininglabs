package global.skymind.solution.datavec;


import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.arbiter.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.IteratorDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class LoadCSV {
    private static int numLinesToSkip = 0;
    private static char delimiter = ',';

    private static int batchSize = 150; // Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)
    private static int labelIndex = 4; // index of label/class column
    private static int numClasses = 3; // number of class in iris dataset

    public static void main(String[] args) throws Exception {
        // define csv file location
        File inputFile = new ClassPathResource("datavec/iris.txt").getFile();
        FileSplit fileSplit = new FileSplit(inputFile);

        // get dataset using record reader. CSVRecordReader handles loading/parsing
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReader.initialize(fileSplit);

        // create iterator from record reader
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);
        DataSet allData = iterator.next();

        System.out.println("Shape of allData vector:");
        System.out.println(Arrays.toString(allData.getFeatures().shape()));

        // shuffle and split all data into training and test set
        allData.shuffle();
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.8);
        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        System.out.println("\nShape of training vector:");
        System.out.println(Arrays.toString(trainingData.getFeatures().shape()));
        System.out.println("\nShape of test vector:");
        System.out.println(Arrays.toString(testData.getFeatures().shape()));

        // create iterator for splitted training and test dataset
        DataSetIterator trainIterator = new IteratorDataSetIterator(trainingData.iterator(), 16);
        DataSetIterator testIterator = new IteratorDataSetIterator(testData.iterator(), 16);

        System.out.println("\nShape of training batch vector:");
        System.out.println(Arrays.toString(trainIterator.next().getFeatures().shape()));
        System.out.println("\nShape of test batch vector:");
        System.out.println(Arrays.toString(trainIterator.next().getFeatures().shape()));
    }
}
