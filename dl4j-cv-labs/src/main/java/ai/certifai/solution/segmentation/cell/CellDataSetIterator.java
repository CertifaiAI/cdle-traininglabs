/*
 * Copyright (c) 2019 Skymind AI Bhd.
 * Copyright (c) 2020 CertifAI Sdn. Bhd.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.certifai.solution.segmentation.cell;

import ai.certifai.solution.segmentation.imageUtils.CustomLabelGenerator;
import ai.certifai.Helper;
import ai.certifai.solution.utilities.DataUtilities;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class CellDataSetIterator {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(CellDataSetIterator.class);

    private static final int height = 224;
    private static final int width = 224;
    private static final int channels = 1;
    private static final long seed = 12345;
    private static final Random random = new Random(seed);
    private static String inputDir;
    private static String downloadLink;
    private static List<Pair<String, String>> replacement = Arrays.asList(
            new org.nd4j.linalg.primitives.Pair<>("inputs", "masks")
    );
    private static CustomLabelGenerator labelMaker = new CustomLabelGenerator(height, width, channels, replacement);
    private static InputSplit trainData, valData;
    private static int batchSize;

    //scale input to 0 - 1
    private static DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    private static ImageTransform transform;

    public CellDataSetIterator() {
    }

    //This method instantiates an ImageRecordReader and subsequently a RecordReaderDataSetIterator based on it
    private static RecordReaderDataSetIterator makeIterator(InputSplit split) throws IOException {
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        //        Both train and val iterator need the preprocessing of converting RGB to Grayscale
        recordReader.initialize(split, transform);
        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, 1, true);
        iter.setPreProcessor(scaler);

        return iter;
    }

    public static RecordReaderDataSetIterator trainIterator() throws IOException {
        return makeIterator(trainData);
    }

    public static RecordReaderDataSetIterator valIterator() throws IOException {
        return makeIterator(valData);
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
            log.info(String.format("Downloading %s from %s.", dataZip.getAbsolutePath(), downloadLink));
            DataUtilities.downloadFile(downloadLink, dataZip.getAbsolutePath());
        }

        if (!classFolder.exists()) {

            if (!Helper.getCheckSum(dataZip.getAbsolutePath())
                    .equalsIgnoreCase(Helper.getPropValues("dataset.segmentationCell.hash"))) {
                System.out.println("Downloaded file is incomplete");
                System.exit(0);
            }

            log.info(String.format("Extracting %s into %s.", dataZip.getAbsolutePath(), classFolder.getAbsolutePath()));
            DataUtilities.extractZip(dataZip.getAbsolutePath(), classFolder.getAbsolutePath());
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
