package ai.certifai.solution.image_processing;/*
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
import static org.bytedeco.opencv.global.opencv_core.BORDER_DEFAULT;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.Laplacian;

import java.io.IOException;

/*
* TASKS:
* -----
* 1. Load an image from the resources/image_processing folder
* 2. Apply Laplacian filter
* 3. Display the image before and after applying Laplacian filter
*
* */

public class LaplacianFilter {
    public static void main(String[] args) throws IOException{
        // Load & Display image
        Mat src = imread(new ClassPathResource("image_processing/lena.png").getFile().getAbsolutePath());
        Display.display(src, "Input");

        // Apply Laplacian filter & display image
        Mat dest = new Mat();
        Laplacian(src, dest, src.depth(),3,3,0, BORDER_DEFAULT);
        Display.display(dest, "Laplacian Filter");

    }

}
