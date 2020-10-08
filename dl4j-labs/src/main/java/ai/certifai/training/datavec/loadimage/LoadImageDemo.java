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

package ai.certifai.training.datavec.loadimage;

import ai.certifai.Helper;
import org.apache.commons.io.FileUtils;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.image.loader.BaseImageLoader;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.util.ArchiveUtils;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Paths;
import java.util.Random;

public class LoadImageDemo {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(LoadImageDemo.class);
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

        /*
        Exercise 2: Create image iterator
        - create FileSplit point to images parent folder
        - create random path filter using RandomPathFilter
        - split images into training and test dataset using FileSplit.sample
        - read image using ImageRecordReader
        - define and initialize image transformation
        - create dataset iterator
        - set image data normalization
         */

        //create FileSplit point to images parent folder
        /*
        YOUR CODE HERE
         */

        //create random path filter using RandomPathFilter
        /*
        YOUR CODE HERE
         */

        //split images into training and test dataset using FileSplit.sample
        /*
        YOUR CODE HERE
         */

        //read image using ImageRecordReader
        /*
        YOUR CODE HERE
         */

        //define and initialize image transformation
        /*
        YOUR CODE HERE
         */

        //create dataset iterator
        /*
        YOUR CODE HERE
         */

        //set image data normalization
        /*
        YOUR CODE HERE
         */

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
