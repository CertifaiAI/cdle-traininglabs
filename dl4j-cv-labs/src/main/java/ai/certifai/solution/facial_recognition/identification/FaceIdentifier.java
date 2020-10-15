/*
 * Copyright (c) 2019 Skymind Holdings Bhd.
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

package ai.certifai.solution.facial_recognition.identification;

import ai.certifai.solution.facial_recognition.detection.FaceLocalization;
import org.bytedeco.opencv.opencv_core.Mat;

import java.io.IOException;
import java.util.List;

public class FaceIdentifier implements IFaceIdentifier{
    public static final String FEATURE_DISTANCE_VGG16_PREBUILT = "FEATURE_DISTANCE_VGG16_PREBUILT";
    public static final String FEATURE_DISTANCE_INCEPTION_RESNET_PREBUILT = "FEATURE_DISTANCE_INCEPTION_RESNET_PREBUILT";

    public List<List<Prediction>> recognize(List<FaceLocalization> faceLocalizations, Mat image) throws IOException {
        return null;
    }
}
