/*
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

import org.bytedeco.opencv.opencv_core.Mat;
import org.nd4j.common.io.ClassPathResource;

import java.io.IOException;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.dilate;
import static org.bytedeco.opencv.global.opencv_imgproc.erode;

/*
 *
 * TASKS:
 * -----
 * 1. Load and display lena.png from the resources/image_processing folder
 * 2. Apply erosion to the image
 * 3. Apply dilation to the image
 * 4. Display the input image, and the two images after applying erosion and dilation separately
 *
 * */

public class ErosionAndDilation {
    public static void main(String[] args) throws IOException {
        // Load and display image from the resources/image_processing folder
        Mat src = imread(new ClassPathResource("image_processing/lena.png").getFile().getAbsolutePath());
        Display.display(src, "Input");

        // Apply erosion and dilation on the loaded image
        Mat eroded = new Mat();
        Mat dilated = new Mat();

        erode(src, eroded, new Mat());
        dilate(src, dilated, new Mat());

        // Display eroded and dilated image
        Display.display(eroded, "eroded");
        Display.display(dilated, "dilated");
    }
}
