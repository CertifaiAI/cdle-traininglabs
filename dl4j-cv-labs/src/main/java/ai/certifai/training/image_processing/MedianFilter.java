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

import global.skymind.solution.image_processing.utils.display.Display;
import org.bytedeco.opencv.opencv_core.Mat;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.IOException;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_GRAYSCALE;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

/*
 * TASKS:
 * -----
 * 1. Load and display x-ray.jpeg from the resources/image_processing folder
 * 2. Add salt and pepper noise to the image
 * 3. Apply median filter on the image which noise is added to denoise it
 * 4. Display the denoised image
 * */

public class MedianFilter {

    public static void main(String[] args) throws IOException {

        // Load Image
        String imgpath = new ClassPathResource("image_processing/x-ray.jpeg").getFile().getAbsolutePath();
        Mat img = imread(imgpath, IMREAD_GRAYSCALE);

        //initialize a random Mat
        Mat randomMat = new Mat(img.rows(), img.cols(), CV_8U);
        randu(randomMat, new Mat(new int[] {0}), new Mat(new int[] {255}));

        //perform thresholding on the random Mat to generate salt and pepper noise
        Mat saltpepperNoise = new Mat();
        threshold(randomMat, saltpepperNoise, 250, 255, THRESH_BINARY);

        //add noise to the original image
        Mat imgNoised = img.clone();
        add(img, saltpepperNoise, imgNoised);

        //perform median filtering on the image with noise
        Mat filteredImg = new Mat();
        medianBlur(imgNoised, filteredImg, 3);

        //display all images
        Display.display(img, "Original");
        Display.display(imgNoised, "Noise");
        Display.display(filteredImg, "Denoised with Median Filter");
    }
}
