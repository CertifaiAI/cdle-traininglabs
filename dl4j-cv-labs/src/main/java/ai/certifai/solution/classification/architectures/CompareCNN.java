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

import ai.certifai.training.classification.WeatherDataSetIterator;
import javafx.util.Pair;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.slf4j.Logger;

import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;

import static ai.certifai.solution.classification.architectures.CreateCNN.*;

public class CompareCNN {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(CompareCNN.class);

    private static final int trainPerc = 80;
    private static final int batchSize = 16;
    private static DataSetIterator trainIter;
    private static DataSetIterator testIter;

    public static void main(String[] args) throws IOException, IllegalAccessException {

        // Prepare data.
        WeatherDataSetIterator.setup(batchSize, trainPerc);
        trainIter = WeatherDataSetIterator.trainIterator();
        testIter = WeatherDataSetIterator.testIterator();

        // Create configuration for model training.
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                            .updater(new Nesterovs(5e-5))
                            .seed(123)
                            .build();

        // Build 3 models based on different CNN architectures, keep them in an iterator containing pairs of name:computationgraph.
        Iterator<Pair<String, ComputationGraph>> models = Arrays.asList(
                        new Pair<>("VGG16", buildVGG16(fineTuneConf)),
                        new Pair<>("VGG19", buildVGG19(fineTuneConf)),
                        new Pair<>("SqueezeNet", buildSqueezeNet(fineTuneConf))
        ).iterator();


        // Run experiment: train the 3 models built and compare their evaluation scores.
        while (models.hasNext()) {
            Pair<String, ComputationGraph> model = models.next();
            ComputationGraph graph = model.getValue();
            String name = model.getKey();
            System.out.println( name + " MODEL TRAINING STARTS" + "\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" );
            System.out.println(graph.summary());

            // Visualize network training using dashboard and set up listener to capture information during training.
            getListeners(graph);
            graph.fit(trainIter, 1);
            Evaluation eval = graph.evaluate(testIter);
            System.out.println(name + "\n" + eval.stats());
            log.info(name + "\n" + eval.stats());
            System.out.println( name + " MODEL TRAINING ENDS" + "\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n\n\n" );
            trainIter.reset();
            testIter.reset();
        }
    }
    private static void getListeners(ComputationGraph model){
        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);
        model.setListeners(
                new StatsListener(storage),
                new ScoreIterationListener(1),
                new EvaluativeListener(trainIter, 1, InvocationType.EPOCH_END),
                new EvaluativeListener(testIter, 1, InvocationType.EPOCH_END));
    }
}
