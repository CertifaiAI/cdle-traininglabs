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

package ai.certifai.solution.image_processing;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.nd4j.common.io.ClassPathResource;

import java.io.IOException;

import static org.bytedeco.opencv.global.opencv_core.add;
import static org.bytedeco.opencv.global.opencv_core.subtract;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.GaussianBlur;

/*
 * TASKS:
 * -------
 * 1. Load any image from the Resources folder
 * 2. Apply Unsharp Masking by following the steps shown in the Day 6 lecture slides on Unsharp Masking
 * 3. Display the images of "before" and "after" Unsharp Masking
 *
 * */

public class UnsharpMasking {
    public static void main(String[] args) throws IOException {
        // Load image
        String imgpath = new ClassPathResource("image_processing/lena.png").getFile().getAbsolutePath();
        Mat src = imread(imgpath);

        // smoothed = GaussianBlur(src)
        Mat smoothed = new Mat();
        GaussianBlur(src, smoothed, new Size(3, 3), 2);

        // detail = src - smoothed
        Mat detail = new Mat();
        subtract(src, smoothed, detail);

        // sharpened = src + detail
        Mat sharpened = new Mat();
        add(src, detail, sharpened);

        // Display the input, detail and sharpened image
        Display.display(src, "Original");
        Display.display(detail, "Detail");
        Display.display(sharpened, "Sharpened");
    }
}
