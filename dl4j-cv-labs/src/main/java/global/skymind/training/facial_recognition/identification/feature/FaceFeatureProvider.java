/*
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

package global.skymind.training.facial_recognition.identification.feature;

import global.skymind.training.facial_recognition.detection.FaceLocalization;
import global.skymind.training.facial_recognition.identification.Prediction;
import org.bytedeco.opencv.opencv_core.Mat;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class FaceFeatureProvider implements IFaceFeatureProvider {

    @Override
    public INDArray getEmbeddings(INDArray arr) {
        return null;
    }

    @Override
    public ArrayList<LabelFeaturePair> setupAnchor(File classDict) throws IOException, ClassNotFoundException {
        return null;
    }

    public List<Prediction> predict(Mat image, FaceLocalization faceLocalization, double threshold, int numSamples) throws IOException {
            return null;
    }

    public int decodeLabelID(INDArray encoded) {
        int topX;
        topX = Nd4j.argMax(encoded.getRow(0).dup(), 1).getInt(0);
        encoded.getRow(0).dup().putScalar(0, topX, 0.0D);
        return topX;
    }
}
