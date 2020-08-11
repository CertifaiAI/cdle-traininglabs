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

package ai.certifai.training.facial_recognition.identification;

import ai.certifai.training.facial_recognition.identification.feature.FaceFeatureProvider;
import ai.certifai.training.facial_recognition.detection.FaceLocalization;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DistanceFaceIdentifier extends FaceIdentifier {
    private static final Logger log = LoggerFactory.getLogger(DistanceFaceIdentifier.class);
    private final FaceFeatureProvider _FaceFeatureProvider;
    private final double threshold;
    private final int numSamples;

     public DistanceFaceIdentifier(FaceFeatureProvider faceFeatureProvider, File dictDir, double threshold, int numSamples) throws IOException, ClassNotFoundException {

            this._FaceFeatureProvider = faceFeatureProvider;
        _FaceFeatureProvider.setupAnchor(dictDir);
        this.threshold = threshold;
        this.numSamples = numSamples;
    }

    @Override
    public List<List<Prediction>> recognize(List<FaceLocalization> faceLocalizations, Mat image) throws IOException {
        List<List<Prediction>> collection = new ArrayList<>();
        for (int i = 0; i<faceLocalizations.size(); i++) {

            int X = (int) faceLocalizations.get(i).getLeft_x();
            int Y = (int) faceLocalizations.get(i).getLeft_y();
            int Width = faceLocalizations.get(i).getValidWidth(image.size().width());
            int Height = faceLocalizations.get(i).getValidHeight(image.size().height());

            // Crop face, Resize and convert into INDArray
            Mat crop_image = new Mat(image, new Rect(X, Y, Width, Height));

            // Get a collection of predictions
            List<Prediction> predictions = _FaceFeatureProvider.predict(
            crop_image, faceLocalizations.get(i), this.threshold, this.numSamples);

            collection.add(predictions);
        }
        return collection;
    }
}
