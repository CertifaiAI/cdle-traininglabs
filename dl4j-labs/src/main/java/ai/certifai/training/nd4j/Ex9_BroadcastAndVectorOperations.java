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

package ai.certifai.training.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Ex9_BroadcastAndVectorOperations {
    public static final String BLACK_BOLD = "\033[1;30m";
    public static final String BLUE_BOLD = "\033[1;34m";
    public static final String ANSI_RESET = "\u001B[0m";

    public static void main(String[] args) {
        int nRows = 3;
        int nColumns = 5;
        INDArray shape = Nd4j.create(new int[]{nRows, nColumns});
        INDArray myArray = Nd4j.rand(shape,123);
        System.out.println(BLACK_BOLD + "Default array" + ANSI_RESET);
        System.out.println(myArray);

        //Add row vector
        INDArray rowVector = Nd4j.ones(1,5);
        INDArray addedRowVector = myArray.addRowVector(rowVector);
        System.out.println(BLACK_BOLD + "\nAdd row vector" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.addRowVector(rowVector)" + ANSI_RESET);
        System.out.println(addedRowVector);

        //Subtract column vector
        INDArray columnVector = Nd4j.ones(3,1);
        INDArray subtractedColVector = myArray.subColumnVector(columnVector);
        System.out.println(BLACK_BOLD + "\nAdd row vector" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.subColumnVector(columnVector)" + ANSI_RESET);
        System.out.println(subtractedColVector);

        /*
        EXERCISE:
        - Create arr1 with shape(3,3) initialize with random value
        - Divide arr1 with row vector of [2,2,2]
        - Multiply arr1 with column vector of [1,2,3]
        */

    }
}
