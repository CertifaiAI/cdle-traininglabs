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

package ai.certifai.solution.datavec.loadcsv;

import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.writer.RecordWriter;
import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.split.partition.Partitioner;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.CategoricalColumnCondition;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.integer.ReplaceInvalidWithIntegerTransform;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.common.io.ClassPathResource;

import java.io.File;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class LoadCSVDemo {
    private static File inputFile;
    public static void main(String[] args) throws Exception {
        inputFile = new ClassPathResource("datavec/titanic/train.csv").getFile();

        //Define input reader
        RecordReader rr = new CSVRecordReader(1, ',');
        rr.initialize(new FileSplit(inputFile));

        Schema inputDataSchema = new Schema.Builder()
                .addColumnString("PassengerId")
                .addColumnCategorical("Survived", Arrays.asList("1", "0"))
                .addColumnCategorical("Pclass",Arrays.asList("1","2","3"))
                .addColumnString("Name")
                .addColumnCategorical("Sex","male","female")
                .addColumnInteger("Age")
                .addColumnInteger("SibSp")
                .addColumnInteger("Parch")
                .addColumnString("Ticket")
                .addColumnDouble("Fare")
                .addColumnString("Cabin")
                .addColumnCategorical("Embarked", Arrays.asList("S","C","Q"))
                .build();

        //Print out the schema:
        System.out.println("Input data schema details:");
        System.out.println(inputDataSchema);

        System.out.println("\n\nOther information obtainable from schema:");
        System.out.println("Number of columns: " + inputDataSchema.numColumns());
        System.out.println("Column names: " + inputDataSchema.getColumnNames());
        System.out.println("Column types: " + inputDataSchema.getColumnTypes());

        // cabin has 687 null values
        // embarked has 2 null values
        // age has 177 null values
        // define transform process
        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                .removeColumns("PassengerId","Name","Ticket","Cabin")
                .transform(new ReplaceInvalidWithIntegerTransform("Age", 0))
                .conditionalReplaceValueTransform("Embarked", new Text("S"),
                        new CategoricalColumnCondition("Embarked",ConditionOp.Equal, ""))
                .categoricalToInteger("Sex")
                .categoricalToOneHot("Pclass","Embarked")
                .build();

        Schema outputSchema = tp.getFinalSchema();
        System.out.println("\n\n\nSchema after transforming data:");
        System.out.println(outputSchema);

        //Process the data:
        List<List<Writable>> originalData = new ArrayList<>();
        while(rr.hasNext()){
            originalData.add(rr.next());
        }

        //Apply transform process
        List<List<Writable>> processedData = LocalTransformExecutor.execute(originalData, tp);

        //Create iterator from processedData
        RecordReader collectionRecordReader = new CollectionRecordReader(processedData);
        DataSetIterator iterator = new RecordReaderDataSetIterator(collectionRecordReader,processedData.size(),0,2);

        DataSet allData = iterator.next();

        //Shuffle and split data into training and test dataset
        allData.shuffle();
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.8);
        org.nd4j.linalg.dataset.DataSet trainingData = testAndTrain.getTrain();
        org.nd4j.linalg.dataset.DataSet testData = testAndTrain.getTest();

        //Create iterator for splitted training and test dataset
        DataSetIterator trainIterator = new ViewIterator(trainingData, 4);
        DataSetIterator testIterator = new ViewIterator(testData, 2);

        //Normalize data to 0 - 1
        DataNormalization scaler = new NormalizerMinMaxScaler(0,1);
        scaler.fit(trainIterator);
        trainIterator.setPreProcessor(scaler);
        testIterator.setPreProcessor(scaler);

        System.out.println("Sample of training vector: \n"+ trainIterator.next());

//        writeAndPrint(processedData);
    }

    private static void writeAndPrint(List<List<Writable>> processedData) throws Exception {
        // write processed data into file
        RecordWriter rw = new CSVRecordWriter();
        File outputFile = new File("titanicTransform.csv");
        if(outputFile.exists()) outputFile.delete();
        outputFile.createNewFile();

        Partitioner p = new NumberOfRecordsPartitioner();
        rw.initialize(new FileSplit(outputFile), p);
        rw.writeBatch(processedData);
        rw.close();

        //Print before + after:
        System.out.println("\n\n---- Original Data File ----");
        String originalFileContents = FileUtils.readFileToString(inputFile, Charset.defaultCharset());
        System.out.println(originalFileContents);

        System.out.println("\n\n---- Processed Data File ----");
        String fileContents = FileUtils.readFileToString(outputFile, Charset.defaultCharset());
        System.out.println(fileContents);
    }


}
