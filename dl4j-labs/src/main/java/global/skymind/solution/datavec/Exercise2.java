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

package global.skymind.solution.datavec;

import global.skymind.Helper;
import org.apache.commons.io.FileUtils;
import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.util.ArchiveUtils;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Exercise2 {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(Exercise2.class);
    private static String dataDir;
    private static String downloadLink;

    private static Random randNumGen = new Random(123);
    private static String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

    private static int height = 50;
    private static int width = 50;
    private static int channels = 3;

    private static int batchSize = 24;
    private static int numLabels = 12;

    private static DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

    public static void main(String[] args) throws Exception {
        downloadLink= Helper.getPropValues("dataset.plant.seed.url");;
        dataDir= Paths.get(System.getProperty("user.home"),Helper.getPropValues("dl4j_home.data")).toString();

        File parentDir = new File(Paths.get(dataDir,"plant-seedlings-classification","train").toString());
        if(!parentDir.exists()) downloadAndUnzip();

        //create FileSplit point to images parent folder
        FileSplit fileSplit = new FileSplit(parentDir);

        //create random path filter using RandomPathFilter
        RandomPathFilter pathFilter = new RandomPathFilter(randNumGen, allowedExtensions);

        //split images into training and test dataset using FileSplit.sample
        InputSplit[] filesInDirSplit = fileSplit.sample(pathFilter, 80, 20);
        InputSplit trainData = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1];

        //read image using ImageRecordReader
        ImageRecordReader trainRR = new ImageRecordReader(height,width,channels,labelMaker);
        ImageRecordReader testRR = new ImageRecordReader(height,width,channels,labelMaker);

        //define and initialize image transformation
        FlipImageTransform horizontalFlip = new FlipImageTransform(1);
        ImageTransform cropImage = new CropImageTransform(5);
        ImageTransform rotateImage = new RotateImageTransform(randNumGen, 15);
        boolean shuffle = false;
        List<Pair<ImageTransform,Double>> pipeline = Arrays.asList(
                new Pair<>(horizontalFlip,0.5),
                new Pair<>(rotateImage, 0.5),
                new Pair<>(cropImage,0.3)
        );
        ImageTransform transform = new PipelineImageTransform(pipeline,shuffle);

        trainRR.initialize(trainData,transform);
        testRR.initialize(testData);

        //create dataset iterator
        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, numLabels);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, numLabels);

        //set image data normalization
        trainIter.setPreProcessor(scaler);
        trainIter.setPreProcessor(scaler);

        System.out.println(trainIter.next());
    }

    private static void downloadAndUnzip() throws IOException {
        String dataPath = new File(dataDir).getAbsolutePath();
        File zipFile = new File(dataPath, "plant-seedings-classification.zip");

        log.info("Downloading the dataset from "+downloadLink+ "...");
        FileUtils.copyURLToFile(new URL(downloadLink), zipFile);

        if(!Helper.getCheckSum(zipFile.getAbsolutePath())
                .equalsIgnoreCase(Helper.getPropValues("dataset.plant.seed.hash"))){
            log.info("Downloaded file is incomplete");
            System.exit(0);
        }

        log.info("Unzipping "+zipFile.getAbsolutePath());
        ArchiveUtils.unzipFileTo(zipFile.getAbsolutePath(), dataPath);
    }
}
