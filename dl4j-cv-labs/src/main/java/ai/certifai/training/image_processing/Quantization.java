package ai.certifai.training.image_processing;/*
 *
 *  * ******************************************************************************
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


import java.io.IOException;

/**
 * Quantization is the discretization of the image pixel value.
 * The equation of quantization is given as follows:
 * floor (pixel_value / bin) * bin
 *
 * In this example, we will reduce the intensity of an image from 256 bits to 2 bits.
 *
 * TASKS:
 * -----
 * 1. Load and display lena.png from the resources/image_processing folder
 * 2. Read the image to a mat file
 * 3. Perform depth reduction algorithm using ND4J
 * 4. Display the final image
 */

public class Quantization {
    public static void main(String[] args) throws IOException{

    }
}
