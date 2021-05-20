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

package ai.certifai.training.segmentation.car;

import org.datavec.image.transform.*;

import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;

import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.UNet;
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import org.slf4j.Logger;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_imgproc.CV_RGB2GRAY;

public class PretrainedUNET {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(
            PretrainedUNET.class);
    /*
     * Instructions for this lab exercise:
     * STEP 1: Import pretrained UNET (provided in model zoo).
     * STEP 2: Configuration of transfer learning.
     * STEP 3: Load data into RecordReaderDataSetIterator.
     * STEP 4: Run training.
     * STEP 5: We will use IOU (Intersection Over Union) as our evaluation metric. Complete the code for IOU calculation.
     * STEP 6: Hyperparameter Tuning. (Epochs, learning rate etc.)
     *
     * */

    //  STEP 2: Set Training hyperparameters here.
    /**
     private static final String featurizeExtractionLayer = "conv2d_4";
    private static final long seed = 12345;
    private static final int nEpochs = ;
    private static final int height = ;
    private static final int width = ;
    private static final int batchSize = ;
    private static final double trainPerc = ;
     **/

    private static String modelExportDir;

    public static void main(String[] args) throws IOException {

        ZooModel zooModel = UNet.builder().build();

        ComputationGraph unet = (ComputationGraph) zooModel.initPretrained(PretrainedType.SEGMENT);
        System.out.println(unet.summary());


        StatsStorage statsStorage = new InMemoryStatsStorage();
        StatsListener statsListener = new StatsListener(statsStorage);
        ScoreIterationListener scoreIterationListener= new ScoreIterationListener(1);

        //STEP 3: Configuration of transfer learning
        /**
        FineTuneConfiguration fineTuneConf = ;

        ComputationGraph unetTransfer = ;
        **/

//        System.out.println(unetTransfer.summary());
//        unetTransfer.setListeners(statsListener, scoreIterationListener);

        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();
        uiServer.attach(statsStorage);

        // STEP 4: Load data into RecordReaderDataSetIterator
        /**
        CarDataSetIterator.setup(batchSize, trainPerc, getImageTransform());
         **/

        //Create iterators
        RecordReaderDataSetIterator imageDataSetTrain = CarDataSetIterator.trainIterator();
        RecordReaderDataSetIterator imageDataSetVal = CarDataSetIterator.valIterator();


//        STEP 5: Run Training
//        for(int i=0; i<nEpochs; i++){
//
//            log.info("Epoch: " + i);
//
//            while(imageDataSetTrain.hasNext())
//            {
//                DataSet imageSet = imageDataSetTrain.next();
//
//                unetTransfer.fit(imageSet);
//
//                INDArray predict = unetTransfer.output(imageSet.getFeatures())[0];
//
//                for (int n=0; n<imageSet.asList().size(); n++){
//                    visualisation.visualize(
//                            imageSet.get(n).getFeatures(),
//                            imageSet.get(n).getLabels(),
//                            predict.get(NDArrayIndex.point(n)),
//                            frame,
//                            panel,
//                            4,
//                            224,
//                            224
//                    );
//                }
//            }
//            imageDataSetTrain.reset();
//        }


        // VALIDATION
        Evaluation eval = new Evaluation(2);

        // VISUALISATION -  validation
//        JFrame frameVal = visualisation.initFrame("Viz");
//        JPanel panelVal = visualisation.initPanel(
//                frame,
//                1,
//                height,
//                width,
//                1
//        );



        float IOUtotal = 0;
        int count = 0;
        while(imageDataSetVal.hasNext()) {
            DataSet imageSetVal = imageDataSetVal.next();

//            INDArray predict = unetTransfer.output(imageSetVal.getFeatures())[0];
            INDArray labels = imageSetVal.getLabels();

            count++;

//            eval.eval(labels, predict);

            log.info(eval.stats());

            //STEP 4: Complete the code for IOU calculation here
            //Intersection over Union:  TP / (TP + FN + FP)
            /**
            float IOUCar = ;
            **/

        }
        System.out.print("Mean IOU: " + IOUtotal/count);

    }


    public static ImageTransform getImageTransform() {
        ImageTransform rgb2gray = new ColorConversionTransform(CV_RGB2GRAY);

        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(
                new Pair<>(rgb2gray, 1.0)
        );
        return new PipelineImageTransform(pipeline, false);
    }

}