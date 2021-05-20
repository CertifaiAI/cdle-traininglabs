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

package ai.certifai.solution.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Ex5_ScalarOperations {
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

        //add operation
        INDArray addArray = myArray.add(1);
        System.out.println(BLACK_BOLD + "\nAdd 1 to array" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.add(1)" + ANSI_RESET);
        System.out.println(addArray);

        //add operation inplace
        myArray.addi(1);
        System.out.println(BLACK_BOLD + "\nAdd 1 to array inplace" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.addi(1)" + ANSI_RESET);
        System.out.println(myArray);

        //add array to array
        INDArray randomArray = Nd4j.rand(3,5);
        INDArray addArraytoArray = myArray.add(randomArray);
        System.out.println(BLACK_BOLD + "\nAdd random array to array (array have to be in same size)" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.add(randomArray)" + ANSI_RESET);
        System.out.println(addArraytoArray);

        /*
        EXERCISE:
        - Create arr1 with shape(3,3) initialize with random value
        - Multiply each of the element on the array with 2
        - Subtract arr1 with arr2. (arr2 = shape(3,3) with value of ones)
        */
        System.out.println(BLACK_BOLD +"\nCreate arr1 with shape(3,3) initialize with random value" + ANSI_RESET);
        INDArray arr1 = Nd4j.randn(3,3);
        System.out.println(arr1);

        System.out.println(BLACK_BOLD +"\nMultiply each of the element on the array with 2" + ANSI_RESET);
        arr1.muli(2);
        System.out.println(arr1);

        System.out.println(BLACK_BOLD +"\nSubtract arr1 with arr2. (arr2 = shape(3,3) with value of ones)" + ANSI_RESET);
        arr1.subi(Nd4j.ones(3,3));
        System.out.println(arr1);
    }
}
