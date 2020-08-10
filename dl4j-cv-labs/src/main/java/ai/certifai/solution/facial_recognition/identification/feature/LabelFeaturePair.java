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

package ai.certifai.solution.facial_recognition.identification.feature;

import org.nd4j.linalg.api.ndarray.INDArray;

public class LabelFeaturePair {
    private final String label;
    private final INDArray embedding;

    public LabelFeaturePair(String label, INDArray embedding) {
        this.label = label;
        this.embedding = embedding;
    }

    public INDArray getEmbedding() {
        return this.embedding;
    }

    public String getLabel() {
        return this.label;
    }
}