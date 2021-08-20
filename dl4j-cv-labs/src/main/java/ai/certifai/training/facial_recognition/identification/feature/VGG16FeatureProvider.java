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

package ai.certifai.training.facial_recognition.identification.feature;

import ai.certifai.training.facial_recognition.detection.FaceLocalization;
import ai.certifai.training.facial_recognition.detection.OpenCV_DeepLearningFaceDetector;
import ai.certifai.training.facial_recognition.identification.Prediction;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import static org.nd4j.linalg.ops.transforms.Transforms.cosineSim;

public class VGG16FeatureProvider extends FaceFeatureProvider {
    private static final Logger log = LoggerFactory.getLogger(VGG16FeatureProvider.class);
    private ComputationGraph model;
    private static ArrayList<LabelFeaturePair> labelFeaturePairList = new ArrayList<>();

    public VGG16FeatureProvider() throws IOException {
        model = (ComputationGraph) VGG16.builder().build().initPretrained(PretrainedType.VGGFACE);
        log.info(model.summary());
    }

    public ArrayList<LabelFeaturePair> setupAnchor(File dictionary) throws IOException, ClassNotFoundException {
        //  This method will crop the face for user and get embedding of the face
        File[] dir = dictionary.listFiles();
        NativeImageLoader nativeImageLoader = new NativeImageLoader();

        //  Looping through the main folder (eg: FaceDB)
        for (File folder : dir) {
            File[] each_folder = folder.listFiles();
            //  Looping inside the subfolder (eg: AndrewNg)
            for (File each_file : each_folder) {
                //  Get matrix of each image
                Mat image = imread(each_file.getAbsolutePath());
                Mat CloneCopy = new Mat();
                image.copyTo(CloneCopy);
                //  Perform face detection
                OpenCV_DeepLearningFaceDetector FaceDetector = new OpenCV_DeepLearningFaceDetector(300, 300, 0.8);
                FaceDetector.detectFaces(CloneCopy);

                //  Handling potential error
                if (FaceDetector.getFaceLocalization().isEmpty() || FaceDetector.getFaceLocalization().size() == 0) {
                    System.out.println("Unable to detect face, please upload another image with clearer front face!");
                    System.out.println("File at: " + each_file.getPath());
                } else if (FaceDetector.getFaceLocalization().size() > 1) {
                    System.out.println("Multiple face detected, please try upload another image!");
                    System.out.println("File at: " + each_file.getPath());
                } else {
                    for (FaceLocalization i : FaceDetector.getFaceLocalization()) {
                        //  Get the bounding box of the image
                        Mat detectedFace = new Mat(image, new Rect((int) i.getLeft_x(), (int) i.getLeft_y(),
                                i.getValidWidth(image.size().width()), i.getValidHeight(image.size().height())));
                        //  Resize for desired input
                        resize(detectedFace, detectedFace, new Size(224, 224));
                        //  Load into INDArray for getting embedding
                        INDArray arr = nativeImageLoader.asMatrix(detectedFace);
                        INDArray embedding = this.getEmbeddings(arr);
                        //  Getting the label of each image by returning parent path name
                        String label = new ParentPathLabelGenerator().getLabelForPath(each_file.getAbsolutePath()).toString();
                        //  Storing the embedding of the face with its respective label
                        labelFeaturePairList.add(new LabelFeaturePair(label, embedding));
                    }
                }
            }
        }
        return labelFeaturePairList;
    }

    public INDArray getEmbeddings(INDArray arr) {
        VGG16ImagePreProcessor _VGG16ImagePreProcessor = new VGG16ImagePreProcessor();
        _VGG16ImagePreProcessor.preProcess(arr);
        return model.feedForward(arr, false).get("fc8");
    }

    public static IntStream reverseOrderStream(IntStream intStream) {
        int[] tempArray = intStream.toArray();
        return IntStream.range(1, tempArray.length + 1).boxed()
                .mapToInt(i -> tempArray[tempArray.length - i]);
    }

    public List<Prediction> predict(Mat image, FaceLocalization faceLocalization, double threshold, int numSamples) throws IOException {
        NativeImageLoader nativeImageLoader = new NativeImageLoader();
        resize(image, image, new Size(224, 224));
        INDArray _image = nativeImageLoader.asMatrix(image);
        INDArray anchor = getEmbeddings(_image);
        List<Prediction> predicted = new ArrayList<>();
        for (LabelFeaturePair i : labelFeaturePairList) {
            INDArray embed = i.getEmbedding();
            double distance = cosineSim(anchor, embed);
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
            if (topNAvg >= threshold) {
                summary.add(new Prediction(entry.getKey(), topNAvg, faceLocalization));
            }
        }

        // sort and select top N
        summary.sort(Comparator.comparing(Prediction::getScore));
        Collections.reverse(summary);

        List<Prediction> result = new ArrayList();
        for (int i = 0; i < 1; i++) {
            if (i < summary.size()) {
                result.add(summary.get(i));
            }
        }
        return result;
    }
}
