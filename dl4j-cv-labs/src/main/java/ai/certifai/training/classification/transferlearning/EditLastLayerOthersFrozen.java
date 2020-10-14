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

package ai.certifai.training.classification.transferlearning;

import ai.certifai.training.classification.DogBreedDataSetIterator;
import org.datavec.image.transform.*;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class EditLastLayerOthersFrozen {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(EditLastLayerOthersFrozen.class);

    private static int epochs = 10;
    private static int batchSize = 32;
    private static int seed = 123;
    private static int numClasses = 5;
    private static int trainPerc = 80;

    private static final Random randNumGen = new Random(seed);

    public static void main(String[] args) throws Exception{
        /*
        Initialize image augmentation
        */
        ImageTransform horizontalFlip = new FlipImageTransform(1);
        ImageTransform cropImage = new CropImageTransform(25);
        ImageTransform rotateImage = new RotateImageTransform(randNumGen, 15);
        ImageTransform showImage = new ShowImageTransform("Image",1000);
        boolean shuffle = false;
        List<Pair<ImageTransform,Double>> pipeline = Arrays.asList(
                new Pair<>(horizontalFlip,0.5),
                new Pair<>(rotateImage, 0.5),
                new Pair<>(cropImage,0.3)
//                ,new Pair<>(showImage,1.0) //uncomment this to show transform image
        );

        ImageTransform transform = new PipelineImageTransform(pipeline,shuffle);

        /*
        Initialize dataset and create training and testing dataset iterator
        */
        DogBreedDataSetIterator.setup(batchSize, trainPerc, transform);
        DataSetIterator trainIter = DogBreedDataSetIterator.trainIterator();
        DataSetIterator testIter = DogBreedDataSetIterator.testIterator();

        /*
        Using pre-configured model
        */
        // Loading vgg16 from zooModel
        ZooModel zooModel = VGG16.builder().build();
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();
        log.info(vgg16.summary());

        // Override the setting for all layers that are not "frozen".
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .updater(new Nesterovs(5e-5, 0.9))
                .seed(seed)
                .build();

        //Construct a new model with the intended architecture and print summary
        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor("fc2") //the specified layer and below are "frozen"
                .removeVertexKeepConnections("predictions") //replace the functionality of the final vertex
                .addLayer("predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(4096).nOut(numClasses)
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.SOFTMAX).build(),
                        "fc2")
                .build();
        log.info(vgg16Transfer.summary());


        /*
        Start a dashboard to visualize network training
        Setup listener to capture useful information during training.
        */
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
        uiServer.attach(statsStorage);
        vgg16Transfer.setListeners(
                new StatsListener( statsStorage),
                new ScoreIterationListener(5),
                new EvaluativeListener(trainIter, 1, InvocationType.EPOCH_END),
                new EvaluativeListener(testIter, 1, InvocationType.EPOCH_END)
        );

        /*
        Start training
        */
        vgg16Transfer.fit(trainIter, epochs);
    }
}
