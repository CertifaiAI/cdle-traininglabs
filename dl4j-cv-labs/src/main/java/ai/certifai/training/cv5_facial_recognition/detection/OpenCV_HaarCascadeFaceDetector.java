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

package ai.certifai.training.facial_recognition.detection;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;
import org.nd4j.common.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY;

public class OpenCV_HaarCascadeFaceDetector extends FaceDetector {

    private static final Logger log = LoggerFactory.getLogger(OpenCV_HaarCascadeFaceDetector.class);
    private CascadeClassifier haar_cascade;
    private static Mat gray = new Mat();
    private static Size minSize = new Size(100, 100);
    private static Size maxSize = new Size(1000, 1000);
    private static RectVector faces = new RectVector();

    public OpenCV_HaarCascadeFaceDetector() throws IOException{
        setModel();
    }

    private void setModel() throws IOException{

        String model_path = new ClassPathResource("fdmodel/OpenCVHaarCascadeFaceDetector/haarcascade_frontalface_default.xml").getFile().toString();
        CascadeClassifier face_cascade = new CascadeClassifier(model_path);
        this.haar_cascade = face_cascade;
    }

    @Override
    public void detectFaces(Mat image) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
        haar_cascade.detectMultiScale(gray, faces, 1.3, 5, 0, minSize, maxSize);
    }

    @Override
    public List<FaceLocalization> getFaceLocalization() {
        List<FaceLocalization> faceLocalizations = new ArrayList();
        for (int i = 0; i < faces.size(); i++) {
            Rect face_i = faces.get(i);
            float tx = face_i.x();
            float ty = face_i.y();
            float bx = tx + face_i.width();
            float by = ty + face_i.height();

            faceLocalizations.add(new FaceLocalization(tx, ty, bx, by));
        }
        return faceLocalizations;
    }
}
