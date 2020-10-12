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
package ai.certifai.solution.image_processing;

import global.skymind.solution.image_processing.utils.histogram.Histogram1DJava;
import org.bytedeco.opencv.opencv_core.Mat;
import org.nd4j.common.io.ClassPathResource;

import java.awt.image.BufferedImage;
import java.io.IOException;

import static org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_GRAYSCALE;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.equalizeHist;


/*
 *
 * Apply Histogram Equalization to enhance contrast of an input image
 * TASKS:
 * -----
 * 1. Load x-ray.jpeg from the resources/image_processing folder
 * 2. Apply the Histogram Equalization function provided in JavaCV
 * 3. Display the image both "before" and "after" contrast enhancement
 * 4. You can create the histogram for an image by creating Histogram1DJava object and calling the getHistogramImage
 *    method.
 *
 * */

public class HistogramEqualization {
    public static void main(String[] args) throws IOException {
        String imgpath = new ClassPathResource("image_processing/x-ray.jpeg").getFile().getAbsolutePath();

        Mat src = imread(imgpath, IMREAD_GRAYSCALE);
        Mat dest = new Mat();
        Histogram1DJava h = new Histogram1DJava();

        equalizeHist(src, dest);

        Display.display(src, "Before Histogram Equalization");
        BufferedImage histogramImageBefore = h.getHistogramImage(src);
        Display.display(histogramImageBefore, "Histogram BEFORE equalization");

        Display.display(dest, "After Histogram Equalization");
        BufferedImage histogramImageAfter = h.getHistogramImage(dest);
        Display.display(histogramImageAfter, "Histogram AFTER equalization");

    }

}
