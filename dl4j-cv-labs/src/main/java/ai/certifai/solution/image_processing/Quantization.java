/*
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
package ai.certifai.solution.image_processing;

import global.skymind.solution.image_processing.utils.display.JPanelDisplay;
import org.bytedeco.opencv.opencv_core.Mat;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.IOException;

import static org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_GRAYSCALE;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

/**
 * Quantization is the discretization of the image pixel value.
 * The equation of quantization is given as follows:
 * floor (pixel_value / bin) * bin
 * <p>
 * In this example, we will reduce the intensity of an image from 256 bits to 2 bits.
 * <p>
 * TASKS:
 * -----
 * 1. Load and display lena.png from the resources/image_processing folder
 * 2. Read the image to a mat file
 * 3. Perform depth reduction algorithm using ND4J
 * 4. Display the final image
 */

public class Quantization {
    private static int bit = 2;

    public static void main(String[] args) throws IOException {
        // Load Image
        String imgpath = new ClassPathResource("image_processing/lena.png").getFile().getAbsolutePath();

        //read img to mat
        Mat src = imread(imgpath, IMREAD_GRAYSCALE);
        NativeImageLoader loader = new NativeImageLoader();
        INDArray imgArray = loader.asMatrix(src);
        int width = src.arrayWidth();
        int height = src.arrayHeight();

        //reduce intensity
        INDArray resultImgArray = reduceDepth(imgArray, width, height, bit);
        Display.display(src, "original");
        JPanelDisplay display = new JPanelDisplay(resultImgArray, "" + bit + "-bit image");
        display.display();
    }

    private static INDArray reduceDepth(INDArray imgArray, int width, int height, int bit) {
        int bin = 256 / bit; //take the original max image depth divide by the bit size to calculate the level of pixel to reducr
        INDArray resultImgArray = Nd4j.create(1, 1, width, height);
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                double pixel = imgArray.getDouble(1, 1, i, j);
                double quantizedValue = Math.floor(pixel / bin) * bin;
                resultImgArray.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.point(i), NDArrayIndex.point(j)},
                        quantizedValue);
            }
        }
        return resultImgArray;
    }
}
