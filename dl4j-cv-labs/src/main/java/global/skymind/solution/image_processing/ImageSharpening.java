package global.skymind.solution.image_processing;/*
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


import org.bytedeco.opencv.opencv_core.Mat;
import org.nd4j.linalg.io.ClassPathResource;
import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.filter2D;

import java.io.IOException;

/*
* TASKS:
* -----
* 1. Load and display sat_map3.jpg from the resources/image_processing folder
* 2. Create a sharpening filter using the following values:
*        0  -1   0
*       -1   5  -1
*        0  -1   0
* 3. Apply the filter on to the input image
* 4. Display the sharpened image
*
* */

public class ImageSharpening {
    public static void main(String[] args) throws IOException {
        // load image
        Mat src = imread(new ClassPathResource("image_processing/sat_map3.jpg").getFile().getAbsolutePath());
        Display.display(src, "Original");

        // Create the sharpening filter
        Mat dest = new Mat();

        int data[] = new int[]{ 0, -1, 0, -1, 5, -1, 0, -1, 0};
        Mat kernel = new Mat(3, 3,CV_32F);
        kernel.put(new Mat(data));

        // Filter and display the image using the kernel created above
        filter2D(src, dest, src.depth(), kernel);
        Display.display(dest, "Sharpened");

    }
}
