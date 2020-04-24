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

package global.skymind.training.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class Ex2_InspectingINDArrays {
    public static final String BLACK_BOLD = "\033[1;30m";
    public static final String BLUE_BOLD = "\033[1;34m";
    public static final String ANSI_RESET = "\u001B[0m";

    public static void main(String[] args) {
        int nRows = 3;
        int nColumns = 5;
        INDArray myArray = Nd4j.rand(nRows, nColumns);

        //Basic information about the array:
        System.out.println(BLACK_BOLD + "Basic INDArray information:" + ANSI_RESET);
        System.out.println("Num. Rows:                 " + myArray.rows());
        System.out.println("Num. Columns:              " + myArray.columns());
        System.out.println("Num. Dimensions / rank:    " + myArray.rank());                    //2 dimensions -> rank 2
        System.out.println("Shape:                     " + Arrays.toString(myArray.shape()));  //[3,5] -> 3 rows, 5 columns
        System.out.println("Length:                    " + myArray.length());
        System.out.println("Datatype:                  " + myArray.data().dataType().name());// 3 rows * 5 columns = 15 total elements

        //Print the array:
        System.out.println(BLACK_BOLD + "\nArray Contents:\n"  + ANSI_RESET + myArray);

        //There are some other ways we can get the same or similar info
        System.out.println();
        System.out.println(BLACK_BOLD + "Others:" + ANSI_RESET);
        System.out.println("size(0) / number of row:       " + myArray.size(0));        //Also equivalent to: .shape()[0]
        System.out.println("size(1) / number of column:    " + myArray.size(1));        //Also equivalent to: .shape()[1]
        System.out.println("Is a vector:                   " + myArray.isVector());
        System.out.println("Is a scalar:                   " + myArray.isScalar());
        System.out.println("Is a matrix:                   " + myArray.isMatrix());
        System.out.println("Is a square matrix:            " + myArray.isSquare());


        /*
        EXERCISE:
        - Create the following array: Nd4j.randn(2,5,6,3,5);
        - Get the shape of the array
        - Get number of dimension
        - Get the size of dimension 4
        */

    }
}
