/*
 * Copyright (c) 2020-2021 CertifAI Sdn. Bhd.
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
 *
 */
package ai.certifai.solution.classification;

import ai.certifai.Helper;
import org.apache.commons.io.FileUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.common.util.ArchiveUtils;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.slf4j.Logger;
import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Paths;
import java.util.Random;

public class WeatherDataSetIterator {

    private static final Logger log = org.slf4j.LoggerFactory.getLogger(WeatherDataSetIterator.class);

    private static final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static Random rngseed = new Random(123);
    private static String dataDir;
    private static String downloadLink;

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

        dataDir = Paths.get(
                System.getProperty("user.home"),
                Helper.getPropValues("dl4j_home.data")
        ).toString();
        downloadLink = Helper.getPropValues("dataset.weather.url");


        File parentDir = new File(Paths.get(dataDir,"WeatherImage").toString());
        if(!parentDir.exists()){
            downloadAndUnzip();
        }

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

    private static void downloadAndUnzip() throws IOException {
        String dataPath = new File(dataDir).getAbsolutePath();
        File zipFile = new File(dataPath, "WeatherImage.zip");

        log.info("Downloading the dataset from "+downloadLink+ "...");
        FileUtils.copyURLToFile(new URL(downloadLink), zipFile);

        if(!Helper.getCheckSum(zipFile.getAbsolutePath())
                .equalsIgnoreCase(Helper.getPropValues("dataset.weather.hash"))){
            log.info("Downloaded file is incomplete");
            System.exit(0);
        }

        log.info("Unzipping "+zipFile.getAbsolutePath());
        ArchiveUtils.unzipFileTo(zipFile.getAbsolutePath(), dataPath);
    }

}
