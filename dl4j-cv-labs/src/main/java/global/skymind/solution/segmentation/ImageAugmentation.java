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

package global.skymind.solution.segmentation;

import global.skymind.Helper;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.datavec.image.data.ImageWritable;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.datavec.image.transform.RotateImageTransform;
import org.nd4j.linalg.primitives.Pair;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class ImageAugmentation {
    protected static final long seed = 12345;
    private static final Random random = new Random(seed);
    private static String inputDir;

    public static void main(String[] args) throws IOException{
        /*
         * This is OPTIONAL. Only run when you want to generate more samples
         *
         * */
        inputDir = Paths.get(
                System.getProperty("user.home"),
                Helper.getPropValues("dl4j_home.data")
        ).toString();

        File imagesPath = new File(Paths.get(inputDir, "data-science-bowl-2018","data-science-bowl-2018","data-science-bowl-2018-2","train","inputs").toString());
        File[] files = imagesPath.listFiles();

        // Set the types of image transformation here.
        ImageTransform flip = new FlipImageTransform();
        ImageTransform rotate = new RotateImageTransform(random, 30);

        List<Pair<ImageTransform, Double>> listOfTransform = Arrays.asList(
                new Pair<>(rotate, 1.0),
                new Pair<>(flip, 0.7)
        );

        PipelineImageTransform transformPipeline = new PipelineImageTransform(listOfTransform, false);

        // Write augmented images to a new folder
        NativeImageLoader niLoader= new NativeImageLoader(224,224,1,flip);

        File augmentedImgFolder = new File(Paths.get(inputDir, "data-science-bowl-2018","data-science-bowl-2018","data-science-bowl-2018-2","train","augmented_inputs").toString());

        if (!augmentedImgFolder.exists() ) {
            augmentedImgFolder.mkdir();
        }

        if (files != null) {
            for (File f : files){

                // ImageWritable -> Frame -> BufferedImage -> png
                ImageWritable iw = niLoader.asWritable(f);
                ImageWritable transformed = transformPipeline.transform(iw);

                Frame frame = transformed.getFrame() ;
                Java2DFrameConverter converter = new Java2DFrameConverter();
                BufferedImage bimage = converter.convert(frame);

                File augmentedImgPath = new File(Paths.get(inputDir, "data-science-bowl-2018","data-science-bowl-2018","data-science-bowl-2018-2","train","augmented_inputs", f.getName()).toString());
                ImageIO.write(bimage, "jpg", augmentedImgPath);
            }
        }


    }

}


