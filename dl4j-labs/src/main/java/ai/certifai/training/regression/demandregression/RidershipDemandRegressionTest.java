/*
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

package ai.certifai.training.regression.demandregression;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

public class RidershipDemandRegressionTest {
    public static void main (String[] args ) throws IOException, InterruptedException {

        File modelSave =  new File(System.getProperty("java.io.tmpdir"), "/trained_regression_model.zip");
        MultiLayerNetwork net = ModelSerializer.restoreMultiLayerNetwork(modelSave);

        File inputFile = new File(System.getProperty("user.home"), ".deeplearning4j/data/ridership/test/test.csv");
        CSVRecordReader csvRR = new CSVRecordReader(1,',');
        csvRR.initialize(new FileSplit(inputFile));

        Schema inputDataSchema = new Schema.Builder()
                .addColumnString("geohash6")
                .addColumnInteger("day")
                .addColumnString("timestamp")
                .addColumnFloat("demand")
                .build();

        Pattern REPLACE_PATTERN = Pattern.compile("\\:\\d+");

        Map<String,String> map = new HashMap<>();
        map.put(REPLACE_PATTERN.toString(), "");

        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                .replaceStringTransform("timestamp", map)
                .convertToInteger("timestamp")
                .transform(new GeohashtoLatLonTransform.Builder("geohash6")
                        .addLatDerivedColumn("latitude")
                        .addLonDerivedColumn("longitude").build())
                .removeColumns("geohash6")
                .build();

        List<List<Writable>> testData = new ArrayList<>();

        while(csvRR.hasNext()){
            testData.add(csvRR.next());
        }

        List<List<Writable>> processedDataTest = LocalTransformExecutor.execute(testData, tp);
        RecordReader collectionRecordReaderTest = new CollectionRecordReader(processedDataTest);

        DataSetIterator testIter = new RecordReaderDataSetIterator(collectionRecordReaderTest, processedDataTest.size(),4,4,true);

        RegressionEvaluation eval = net.evaluateRegression(testIter);
        System.out.println(eval.stats());
    }
}
