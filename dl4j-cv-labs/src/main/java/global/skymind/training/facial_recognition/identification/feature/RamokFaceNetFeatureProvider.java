package global.skymind.training.facial_recognition.identification.feature;

import global.skymind.training.facial_recognition.detection.FaceLocalization;
import global.skymind.training.facial_recognition.identification.Prediction;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import static org.nd4j.linalg.ops.transforms.Transforms.euclideanDistance;

/**
 * generates embedding based on pre-built model, inspiration and reference https://github.com/klevis/Java-Machine-Learning-for-Computer-Vision/tree/master/FaceRecognition
 */

public class RamokFaceNetFeatureProvider extends FaceFeatureProvider {
    private ComputationGraph genEmbd_model;
    private static ArrayList<LabelFeaturePair> labelFeaturePairList = new ArrayList<>();
    private static final Logger log = LoggerFactory.getLogger(RamokFaceNetFeatureProvider.class);


    public RamokFaceNetFeatureProvider() throws IOException {

        ComputationGraph model = ModelSerializer.restoreComputationGraph(new ClassPathResource("EmbeddingGenerator/EmbeddingGenerator.zip").getFile(),false);

        genEmbd_model = new TransferLearning.GraphBuilder(model)
                .setFeatureExtractor("embeddings") // the L2Normalize vertex and layers below are frozen
                .removeVertexAndConnections("lossLayer")
                .setOutputs("embeddings")
                .build();
        genEmbd_model.init();
        System.out.println(genEmbd_model.summary());
    }

    public ArrayList<LabelFeaturePair> setupAnchor(File dictionary) throws IOException {
        ImageRecordReader recordReader = new ImageRecordReader(96, 96, 3, new ParentPathLabelGenerator());
        recordReader.initialize(new FileSplit(dictionary));
        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(recordReader, 1, 1, dictionary.listFiles().length);
        List<String> labels = iter.getLabels();

        generateEmbeddings(iter, labels);

        return labelFeaturePairList;
    }

    private void generateEmbeddings(RecordReaderDataSetIterator iter, List<String> labels) {
        while (iter.hasNext()) {
            DataSet Ds = iter.next();
            INDArray embedding = this.getEmbeddings(Ds.getFeatures());
            String label = labels.get(decodeLabelID(Ds.getLabels()));
            labelFeaturePairList.add(new LabelFeaturePair(label, embedding));
        }
    }
//    Method to generate embeddings from a INDArray
    public INDArray getEmbeddings(INDArray arr) {
        Map<String, INDArray> output  = genEmbd_model.feedForward(normalize(arr),false);
        GraphVertex embeddings = genEmbd_model.getVertex("embeddings");
        INDArray dense = output.get("dense");
        embeddings.setInputs(dense);
        INDArray embeddingValues = embeddings.doForward(false, LayerWorkspaceMgr.builder().defaultNoWorkspace().build());
        return embeddingValues;
    }

    public static IntStream reverseOrderStream(IntStream intStream) {
        int[] tempArray = intStream.toArray();
        return IntStream.range(1, tempArray.length + 1).boxed()
                .mapToInt(i -> tempArray[tempArray.length - i]);
    }

    public List<Prediction> predict(Mat image, FaceLocalization faceLocalization, int numPredictions, double threshold, int numSamples) throws IOException {
        NativeImageLoader nativeImageLoader = new NativeImageLoader();
        resize(image, image, new Size(96, 96));
        INDArray _image = nativeImageLoader.asMatrix(image);
        INDArray anchor = getEmbeddings(_image);
        List<Prediction> predicted = new ArrayList<>();
        for (LabelFeaturePair i: labelFeaturePairList){
            INDArray embed = i.getEmbedding();
//            Switch between methods to calculate distance, Cosine Similarity or Euclidean Distance
//            double distance = cosineSim(anchor, embed);
            double distance = 1-euclideanDistance(anchor, embed);
            predicted.add(new Prediction(i.getLabel(), distance, faceLocalization));
        }

        // aggregator - average comparison per class
        List<Prediction> summary = new ArrayList<>();
        final Map<String, List<Prediction>> map = predicted.stream().collect(Collectors.groupingBy(p -> p.getLabel()));
        for (final Map.Entry<String, List<Prediction>> entry : map.entrySet()) {

            double topNAvg = reverseOrderStream(entry
                    .getValue()
                    .stream()
                    .mapToInt(p -> (int) (p.getScore() * 10000))
                    .sorted()
            )
                    .limit(numSamples)
                    .mapToDouble(num -> (double) num / 10000)
                    .average()
                    .getAsDouble();
            if(topNAvg >= threshold) {
                summary.add(new Prediction(entry.getKey(), topNAvg, faceLocalization));
            }
        }

        // sort and select top N
        summary.sort(Comparator.comparing(Prediction::getScore));
        Collections.reverse(summary);

        List<Prediction> result = new ArrayList();
        for(int i=0; i<numPredictions; i++){
            if(i<summary.size()) {
                result.add(summary.get(i));
            }
        }
        return result;
    }

    private static INDArray normalize(INDArray read) {
        return read.div(255.0);
    }
}
