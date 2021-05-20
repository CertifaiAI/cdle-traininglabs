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

package ai.certifai.solution.facial_recognition.identification;

import ai.certifai.solution.facial_recognition.detection.FaceLocalization;

public class Prediction {

    private String label;
    private double score;
    private FaceLocalization faceLocalization;

    public Prediction(String label, double score, FaceLocalization faceLocalization) {
        this.label = label;
        this.score = score;
        this.faceLocalization = faceLocalization;
    }

    public Prediction(String label, double percentage) {
        this.label = label;
        this.score = percentage;
    }

    public String getLabel(){
        return this.label;
    }

    public double getScore(){
        return this.score;
    }

    public FaceLocalization getFaceLocalization(){
        return this.faceLocalization;
    }

    public String toString() {
//        return String.format("%s: %.2f ", this.label, this.score);
        return String.format("%s", this.label);
    }
}