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

import global.skymind.solution.image_processing.utils.Display;
import org.bytedeco.opencv.opencv_core.Mat;
import org.nd4j.linalg.io.ClassPathResource;
import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.Sobel;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.filter2D;

import java.io.IOException;

/*
* TASKS:
* -----
* 1. Load and display x-ray.jpeg from the resources/image_processing folder
* 2. Create and apply the vertical operator onto the input image
* 3. Create and apply the horizontal operator onto the input image
* 4. Apply both operator onto the input image (by adding them)
* 5. Display the follwing:
*       - image after applying vertical operator
*       - image after applying horizontal operator
*       - image after combining both operators
*
* */

public class SobelOperator {
    public static void main(String[] args) throws IOException {
        // load image
        Mat src = imread(new ClassPathResource("image_processing/x-ray.jpeg").getFile().getAbsolutePath());
        Display.display(src, "Original");

        // Create the vertical operator
        Mat dest1 = new Mat();
        Sobel(src, dest1, 0, 1, 0, 3, 1, 0, BORDER_DEFAULT);
        Display.display(dest1, "Sobel Operator dx");

        // Create the horizontal operator
        Mat dest2 = new Mat();
        Sobel(src, dest2, 0, 0, 1, 3, 1, 0, BORDER_DEFAULT);
        Display.display(dest2, "Sobel Operator dy");

        // Apply both filters on the image and display it
        Mat dest_sobel = new Mat();
        add(dest1, dest2, dest_sobel);
        Display.display(dest_sobel, "Both direction applied");


//        int data1[] = new int[]{ -1, 0, 1, -2, 0, 2, -1, 0, 1};
//        Mat kernel_v = new Mat(3, 3,CV_32F);
//        kernel_v.put(new Mat(data1));
//
//        int data2[] = new int[]{ -1, -2, -1, 0, 0, 0, 1, 2, 1};
//        Mat kernel_h = new Mat(3, 3,CV_32F);
//        kernel_h.put(new Mat(data2));
//
//        Mat dest_v = new Mat();
//        Mat dest_h = new Mat();
//
//        filter2D(src, dest_v, -1, kernel_v);
//        filter2D(src, dest_h, -1, kernel_h);
//
//        Display.display(dest_v, "Vertical direction");
//        Display.display(dest_h, "Horizontal direction");
//
//        Mat dest_sobel = new Mat();
//        add(dest_v, dest_h, dest_sobel);
//        Display.display(dest_sobel, "Both direction applied");
    }
}
