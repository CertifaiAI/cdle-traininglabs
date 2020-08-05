/*
 *
 *  * ******************************************************************************
 *  *  * Copyright (c) 2019 Skymind AI Bhd.
 *  *  * Copyright (c) 2020 CertifAI Sdn. Bhd.
 *  *  *
 *  *  * This program and the accompanying materials are made available under the
 *  *  * terms of the Apache License, Version 2.0 which is available at
 *  *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *  *
 *  *  * Unless required by applicable law or agreed to in writing, software
 *  *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  *  * License for the specific language governing permissions and limitations
 *  *  * under the License.
 *  *  *
 *  *  * SPDX-License-Identifier: Apache-2.0
 *  *  *****************************************************************************
 *
 *
 */

package ai.certifai.training.datavec.loadimage;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class LoadImageAndTransform {
    //Images are of format given by allowedExtension
    private static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;

    private static final long seed = 12345;

    private static final Random randNumGen = new Random(seed);

    private static final int height = 50;
    private static final int width = 50;
    private static final int channels = 3;

    public static void main(String[] args) throws Exception {
        //DIRECTORY STRUCTURE:
        //Images in the dataset have to be organized in directories by class/label.
        //In this example there are ten images in three classes
        //Here is the directory structure
        //                                    parentDir
        //                                  /    |     \
        //                                 /     |      \
        //                            labelA  labelB   labelC
        //Set your data up like this so that labels from each label/class live in their own directory
        //And these label/class directories live together in the parent directory

        // define image folder location
        File parentDir = new ClassPathResource("datavec/flowers").getFile();

        //Files in directories under the parent dir that have "allowed extensions" split needs a random number generator for reproducibility when splitting the files into train and test
        FileSplit fileSplit = new FileSplit(parentDir, allowedExtensions, randNumGen);

        //You do not have to manually specify labels. This class (instantiated as below) will
        //parse the parent dir and use the name of the subdirectories as label/class names
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        //The balanced path filter gives you fine tune control of the min/max cases to load for each class
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);

        //Split the image files into train and test. Specify the train test split as 80%,20%
        InputSplit[] filesInDirSplit = fileSplit.sample(pathFilter, 80, 20);
        InputSplit trainData = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1];

        //Specifying a new record reader with the height and width you want the images to be resized to.
        //Note that the images in this example are all of different size
        //They will all be resized to the height and width specified below
        ImageRecordReader trainRR = new ImageRecordReader(height,width,channels,labelMaker);
        ImageRecordReader testRR = new ImageRecordReader(height,width,channels,labelMaker);

        //Define image transformation
        FlipImageTransform horizontalFlip = new FlipImageTransform(1);
        ImageTransform cropImage = new CropImageTransform(5);
        ImageTransform rotateImage = new RotateImageTransform(randNumGen, 15);
        ImageTransform showImage = new ShowImageTransform("Image",1000);
        boolean shuffle = false;
        List<Pair<ImageTransform,Double>> pipeline = Arrays.asList(
                new Pair<>(horizontalFlip,0.5),
                new Pair<>(rotateImage, 0.5),
                new Pair<>(cropImage,0.3),
                new Pair<>(showImage,1.0)
        );
        ImageTransform transform = new PipelineImageTransform(pipeline,shuffle);

        //Initialize the record reader with the train data and the transform chain
        trainRR.initialize(trainData,transform);
        testRR.initialize(testData);

        int numLabels = trainRR.numLabels();
        int batchSize = 3; // Minibatch size. Here: The number of images to fetch for each call to dataIter.next().
        int labelIndex = 1; // Index of the label Writable (usually an IntWritable), as obtained by recordReader.next()
        // List<Writable> lw = recordReader.next();
        // then lw[0] =  NDArray shaped [1,3,50,50] (1, channels, height, width)
        //      lw[1] =  label as integer.

        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, labelIndex, numLabels);

        int batchIndex = 0;
        while (trainIter.hasNext()) {
            DataSet ds = trainIter.next();

            batchIndex += 1;
            System.out.println("\nBatch number: " + batchIndex);
            System.out.println("Feature vector shape: " + Arrays.toString(ds.getFeatures().shape()));
            System.out.println("Label vector shape: " +Arrays.toString(ds.getLabels().shape()));
        }
    }
}
