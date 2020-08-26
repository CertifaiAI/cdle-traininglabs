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

package ai.certifai.solution.datavec.kfold;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.KFoldIterator;
import org.nd4j.linalg.io.ClassPathResource;


import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/*
 *
 * TASKS:
 * -----
 * 1. Load data using getDataSet() method.
 * 2. create a kFoldIterator object. (set k=5)
 * 3. loop through the kFoldIterator and print out the observations for each training set and test set.
 *
 * */

public class SimpleKFoldDemo {

    public static final String BLACK_BOLD = "\033[1;30m";
    public static final String BLUE_BOLD = "\033[1;34m";
    public static final String ANSI_RESET = "\u001B[0m";

    public static void main(String[] args) throws Exception {
        //Load data using getDataSet() method.
        DataSet dataSet = getDataSet();

        //create KFold Object
        KFoldIterator kFoldIterator = new KFoldIterator(5,dataSet);

        //loop through the kFoldIterator and print out the observations for each training set and test set.
        int i = 1;
        System.out.println("-------------------------------------------------------------");
        while (kFoldIterator.hasNext()){
            System.out.println(BLACK_BOLD + "BATCH: " + i + ANSI_RESET);
            INDArray trainFoldFeatures = kFoldIterator.next().getFeatures();
            INDArray testFoldFeatures = kFoldIterator.testFold().getFeatures();
            System.out.println(BLUE_BOLD + "TRAINING FOLD: \n" + ANSI_RESET);
            System.out.println(trainFoldFeatures);
            System.out.println(BLUE_BOLD + "TEST FOLD: \n" + ANSI_RESET);
            System.out.println(testFoldFeatures);
            i++;
            System.out.println("-------------------------------------------------------------");
        }

    }

    private static DataSet getDataSet() throws IOException, InterruptedException {
        Schema sc = new Schema.Builder()
                .addColumnString("Id")
                .addColumnString("feature 1")
                .addColumnString("feature 2")
                .addColumnString("label")
                .build();

        TransformProcess tp = new TransformProcess.Builder(sc)
                .convertToInteger("Id")
                .convertToInteger("feature 1")
                .convertToInteger("feature 2")
                .convertToInteger("label")
                .build();

        File file = new ClassPathResource("datavec/kfold.csv").getFile();
        RecordReader rr = new CSVRecordReader(1,',');
        rr.initialize(new FileSplit(file));
        List<List<Writable>> originalData = new ArrayList<>();
        while(rr.hasNext()){
            originalData.add(rr.next());
        }
        List<List<Writable>> processedData = LocalTransformExecutor.execute(originalData, tp);
        //Create iterator from processedData
        RecordReader collectionRecordReader = new CollectionRecordReader(processedData);
        DataSetIterator dataSetIter = new RecordReaderDataSetIterator(collectionRecordReader,10,3,3,true);

        return dataSetIter.next();
    }
}
