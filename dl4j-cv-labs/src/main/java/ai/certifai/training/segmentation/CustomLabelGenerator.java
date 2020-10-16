/*
 * Copyright (c) 2019 Skymind AI Bhd.
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

package ai.certifai.training.segmentation;

import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.common.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.List;

public class CustomLabelGenerator implements PathLabelGenerator{
    private static Logger log = LoggerFactory.getLogger(CustomLabelGenerator.class);

    private final int height;
    private final int width;
    private final int channels;
    private final NativeImageLoader imageLoader;
    private static List<Pair<String, String>> replacement;


    //DIRECTORY STRUCTURE:
    //Here is the directory structure
    //                                  parentDir
    //                                 /         \
    //                                /           \
    //                               /             \
    //                           image              mask
    //                          /  |  \            /  |  \
    //                         /   |   \          /   |   \
    //                        /    |    \        /    |    \
    //                   case1  case2  case3   case1 case2  case3

    @Override
    public boolean inferLabelClasses() {
        return false;
    }

    public CustomLabelGenerator(int height, int width, int channels, List replacement ){
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.imageLoader = new NativeImageLoader(this.height, this.width, this.channels);
        this.replacement = replacement;
    }

    // This custom label generator finds labels for each input image by replacing one the folders (inputs >> masks) in the path string.
    @Override
    public Writable getLabelForPath(String path) {

        String labelPath = path;
        try
        {

            for (int i=0; i<replacement.size(); i++) {
                labelPath = labelPath.replace(replacement.get(i).getKey(), replacement.get(i).getValue());
            }

//            System.out.println(labelPath);

            NDArrayWritable label = new NDArrayWritable(imageLoader.asMatrix(new File(labelPath)) );

            INDArray labelINDArray = label.get();

            // normalise to 0-1 scale
            label.set( labelINDArray.div(255));

            return label ;

        }
        catch (IOException ioe)
        {
            ioe.printStackTrace();
            return null;
        }
    }

    @Override
    public Writable getLabelForPath(URI uri) {
        return this.getLabelForPath(uri.getPath());
    }

}

