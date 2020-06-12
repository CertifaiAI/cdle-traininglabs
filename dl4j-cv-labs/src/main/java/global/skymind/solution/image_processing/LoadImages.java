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
import org.bytedeco.opencv.opencv_core.Size;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.IOException;

import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

/*
 * We will be using opencv from bytedeco which is a 3rd party maintained opencv library.
 * Learning the basic of using opencv will help us in developing computer-vision based applications.
 *
 * 1. Go to https://image.online-convert.com/, convert resources/image_processing/opencv.png into the following format:
 *       - .bmp
 *       - .jpg
 *       - .tiff
 *     Save them to the same resources/image_processing folder.
 *
 *  2. Use the .imread function to load each all the images in resources/image_processing,
 *       and display them using Display.display
 *
 *
 *  3. Print the following image attributes:
 *       - depth
 *       - number of channel
 *       - width
 *       - height
 *
 *  4. Repeat step 2 & 3, but this time load the images in grayscale
 *
 *  5. Resize file
 *
 *  6. Write resized file to disk
 *
 * */

public class LoadImages {
    public static void main(String[] args) throws IOException {

        // Load Image
        String imgpath = new ClassPathResource("image_processing/opencv.png").getFile().getAbsolutePath();
        Mat src = imread(imgpath, IMREAD_GRAYSCALE);

        Display.display(src, "Input");

        // Print image attributes
        System.out.println("Number of channel: " + src.channels());
        System.out.println("Image Height: " + src.arrayHeight());
        System.out.println("Image Width： " + src.arrayWidth());

        // Image depth
        /****************************************************
         CV_32F  float
         CV_16S  short
         CV_8S   char   byte
         CV_8U - 8-bit unsigned integers ( 0..255 )    uchar
         CV_8S - 8-bit signed integers ( -128..127 )   byte(java)
         CV_16U - 16-bit unsigned integers ( 0..65535 )
         CV_16S - 16-bit signed integers ( -32768..32767 )
         CV_32S - 32-bit signed integers ( -2147483648..2147483647 )
         CV_32F - 32-bit ﬂoating-point numbers ( -FLT_MAX..FLT_MAX, INF, NAN )
         CV_64F - 64-bit ﬂoating-point numbers ( -DBL_MAX..DBL_MAX, INF, NAN )
         CV_8U=0, CV_8S=1, CV_16U=2, CV_16S=3, CV_32S=4, CV_32F=5, CV_64F=6
         https://github.com/opencv/opencv/blob/a6c02af0991fd359cf4501e02acf5f3f9d4ae91d/modules/core/include/opencv2/core/hal/interface.h#L67
         ***************************************************/
        System.out.println("Image depth: " + src.depth());

        // Image resizing
        Mat dest = new Mat();
        Mat dest_up_linear = new Mat();
        Mat dest_up_nearest = new Mat();
        Mat dest_up_cubic = new Mat();

        // Downsampling
        // Upsampling using diff. interpolation methods
        resize(src, dest, new Size(300, 300)); // DOWNSIZE
        resize(dest, dest_up_linear, new Size(1478, 1200), 0, 0, INTER_LINEAR); //UPSIZE
        resize(dest, dest_up_nearest, new Size(1478, 1200), 0, 0, INTER_NEAREST);
        resize(dest, dest_up_cubic, new Size(1478, 1200), 0, 0, INTER_CUBIC);

        // Display resized images
        Display.display(dest, "Downsized");
        Display.display(dest_up_linear, "Upsized - Linear interpolation");
        Display.display(dest_up_nearest, "Upsized - Nearest Neighbors");
        Display.display(dest_up_cubic, "Upsized - Cubic interpolation");

        // Write image to disk
        String imgsavepath = imgpath.replace("opencv.tiff", "opencv_small.jpg");
        System.out.println(imgsavepath);
        imwrite(imgsavepath, dest);


    }
}
