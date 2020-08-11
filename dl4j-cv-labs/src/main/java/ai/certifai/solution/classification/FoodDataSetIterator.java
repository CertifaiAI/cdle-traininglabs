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

package ai.certifai.solution.classification;

import org.datavec.api.io.filters.BalancedPathFilter;
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
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;


/**
 * Class 0 images: 1350
 * Class 1 images: 150
 */
public class FoodDataSetIterator {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(FoodDataSetIterator.class);

    private static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static final Random rng  = new Random(13);

    private static final int height = 224;
    private static final int width = 224;
    private static final int channels = 3;
    private static final int numClasses = 5;

    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private static DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    private static InputSplit trainData,testData;
    private static int batchSize;

    public static DataSetIterator trainIterator() throws IOException {
        return makeIterator(trainData);

    }

    public static DataSetIterator testIterator() throws IOException {
        return makeIterator(testData);
    }

    public static void setup(int batchSizeArg) throws IOException {
        batchSize = batchSizeArg;

        File trainDir = new File(System.getProperty("user.home"), ".deeplearning4j\\data\\food-dataset\\train");
        FileSplit trainFilesinDir = new FileSplit(trainDir, allowedExtensions, rng);

        File testDir = new File(System.getProperty("user.home"), ".deeplearning4j\\data\\food-dataset\\test");
        FileSplit testFilesinDir = new FileSplit(testDir, allowedExtensions, rng);
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedExtensions, labelMaker);

        System.out.println(trainFilesinDir.length());
        System.out.println(testFilesinDir.length());
        trainData = trainFilesinDir.sample(pathFilter,1)[0];
        testData = testFilesinDir.sample(pathFilter,1)[0];

    }

    private static DataSetIterator makeIterator(InputSplit split) throws IOException {

        ImageTransform flipTransform1 = new FlipImageTransform(rng);
        ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
        ImageTransform rotateTransform = new RotateImageTransform(30);
        boolean shuffle = false;
        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(new Pair<>(flipTransform1, 0.9),
                new Pair<>(flipTransform2, 0.8),
                new Pair<>(rotateTransform, 0.5));

        ImageTransform transform = new PipelineImageTransform(pipeline, shuffle);

        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
        recordReader.initialize(split, transform);

        DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses);
        iter.setPreProcessor(scaler);
        return iter;
    }




}