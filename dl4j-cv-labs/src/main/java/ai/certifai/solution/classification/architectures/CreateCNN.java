package ai.certifai.solution.classification.architectures;/*
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

import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.SqueezeNet;
import org.deeplearning4j.zoo.model.VGG16;
import org.deeplearning4j.zoo.model.VGG19;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;


public class CreateCNN {
    private static int outputNum = 4;
    private static int seed = 423;
    private static LossFunctions.LossFunction lossFunction = LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD;
    private static Activation outputActivation = Activation.SOFTMAX;


    public static ComputationGraph buildVGG19(FineTuneConfiguration fineTuneCOnf) throws IOException {
        ZooModel zooModel = VGG19.builder().build();
        ComputationGraph vgg19 = (ComputationGraph) zooModel.initPretrained();


        return new TransferLearning.GraphBuilder(vgg19)
                    .fineTuneConfiguration(fineTuneCOnf)
                    .setFeatureExtractor("fc2")
                    .removeVertexKeepConnections("predictions")
                    .addLayer("predictions",
                            new OutputLayer.Builder(lossFunction)
                                            .nIn(4096).nOut(outputNum)
                                            .weightInit(WeightInit.XAVIER)
                                            .activation(Activation.SOFTMAX).build(),
                            "fc2")
                    .build();
    }


    public static ComputationGraph buildVGG16(FineTuneConfiguration fineTuneCOnf) throws IOException {
        ZooModel zooModel = VGG16.builder().build();
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();

        return new TransferLearning.GraphBuilder(vgg16)
                .fineTuneConfiguration(fineTuneCOnf)
                .setFeatureExtractor("fc2")
                .removeVertexKeepConnections("predictions")
                .addLayer("predictions",
                        new OutputLayer.Builder(lossFunction)
                                        .nIn(4096).nOut(outputNum)
                                        .weightInit(WeightInit.XAVIER)
                                        .activation(outputActivation).build(),
                        "fc2")
                .build();
    }


    public static ComputationGraph buildSqueezeNet(FineTuneConfiguration fineTuneCOnf) throws IOException {
        ZooModel zooModel3 = SqueezeNet.builder().build();
        ComputationGraph squeezeNet = (ComputationGraph) zooModel3.initPretrained();

        return new TransferLearning.GraphBuilder(squeezeNet)
                        .fineTuneConfiguration(fineTuneCOnf)
                        .setFeatureExtractor("drop9")
                        .removeVertexKeepConnections("conv10")
                        .removeVertexAndConnections("relu10")
                        .removeVertexAndConnections("global_average_pooling2d_5")
                        .removeVertexAndConnections("loss")
                        .addLayer("conv10",
                                new ConvolutionLayer.Builder(1,1).nIn(512).nOut(outputNum)
                                                .build(),
                                "drop9")
                        .addLayer("conv10_act", new ActivationLayer(Activation.RELU), "conv10")
                        .addLayer("global_avg_pool", new GlobalPoolingLayer(PoolingType.AVG), "conv10_act")
                        .addLayer("softmax", new ActivationLayer(outputActivation), "global_avg_pool")
                        .addLayer("loss", new LossLayer.Builder(lossFunction).build(), "softmax")
                        .setOutputs("loss")
                        .build();
    }


}
