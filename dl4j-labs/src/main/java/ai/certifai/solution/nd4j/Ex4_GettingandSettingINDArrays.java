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

package ai.certifai.solution.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class Ex4_GettingandSettingINDArrays {
    public static final String BLACK_BOLD = "\033[1;30m";
    public static final String BLUE_BOLD = "\033[1;34m";
    public static final String ANSI_RESET = "\u001B[0m";

    public static void main(String[] args) {
        int nRows = 3;
        int nColumns = 5;
        INDArray shape = Nd4j.create(new int[]{nRows, nColumns});
        INDArray myArray = Nd4j.rand(shape, 123);
        System.out.println(BLACK_BOLD + "Default array" + ANSI_RESET);
        System.out.println(myArray);

        //Get scalar value
        INDArray scalarVal = myArray.getScalar(0,0); //try getDouble, getInt, and getFloat to see the different
        System.out.println(BLACK_BOLD + "\nGet scalar value, position row 1 and column 1" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.getScalar(0,0)" + ANSI_RESET);
        System.out.println(scalarVal);

        //Set scalar value
        myArray.putScalar(6,0);
        System.out.println(BLACK_BOLD + "\nSet scalar value by index, set value of index 6 to 0" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.getScalar(6,0)" + ANSI_RESET);
        System.out.println(myArray);

        myArray.putScalar(new int[]{1,4},10);
        System.out.println(BLACK_BOLD + "\nSet scalar value based on given location, row 2 and column 5" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.putScalar(new int[]{1,4},10)" + ANSI_RESET);
        System.out.println(myArray);

        //Get row or column
        INDArray row = myArray.getRow(0);
        System.out.println(BLACK_BOLD + "\nGet the first row from myArray" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.getRow(0)" + ANSI_RESET);
        System.out.println(row);

        INDArray rows = myArray.getRows(0,2);
        System.out.println(BLACK_BOLD + "\nGet multiple rows from myArray" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.getRows(0,2)" + ANSI_RESET);
        System.out.println(rows);

        //Set row or column
        myArray.putRow(2, Nd4j.zeros(5));
        System.out.println(BLACK_BOLD + "\nSet row number 2 to zero" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.putRow(2, Nd4j.zeros(5))" + ANSI_RESET);
        System.out.println(myArray);

        //Get an arbitrary sub-arrays based on certain indexes
        INDArray first3Columns = myArray.get(NDArrayIndex.all(), NDArrayIndex.interval(0,3));
        System.out.println(BLACK_BOLD + "\nGet first 3 columns" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.get(NDArrayIndex.all(), NDArrayIndex.interval(0,3))" + ANSI_RESET);
        System.out.println(first3Columns);

        INDArray secondRow = myArray.get(NDArrayIndex.point(1), NDArrayIndex.all());
        System.out.println(BLACK_BOLD + "\nGet second row" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.get(NDArrayIndex.point(1), NDArrayIndex.all())" + ANSI_RESET);
        System.out.println(secondRow);

        INDArray oddColumns = myArray.get(NDArrayIndex.all(), NDArrayIndex.interval(0,2,5));
        System.out.println(BLACK_BOLD + "\nGet all rows and odd column" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.get(NDArrayIndex.all(), NDArrayIndex.interval(0,2,5))" + ANSI_RESET);
        System.out.println(oddColumns);

        //Set an arbitrary sub-arrays based on certain indexes
        myArray.put(
                new INDArrayIndex[]{NDArrayIndex.point(0),NDArrayIndex.all()},
                Nd4j.ones(1,5)
        );
        System.out.println(BLACK_BOLD + "\nSet a first row and all columns to zero" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.put(new INDArrayIndex[]{NDArrayIndex.point(0),NDArrayIndex.all()},Nd4j.zeros(1,5));" + ANSI_RESET);
        System.out.println(myArray);

        /*
        EXERCISE:
        - Create arr1 with shape(3,5) initialize with random value
        - Get value of the 2nd row
        - Change value of first row and second column to 0
        - Replace 2nd row with array of ones
        - Get value of the 3rd row and 2nd - 4th columns
        - Replace the previously selected value with ones
        */
        System.out.println(BLACK_BOLD +"\nCreate arr1 with shape(3,5) initialize with random value" + ANSI_RESET);
        INDArray arr1 = Nd4j.randn(3,5);
        System.out.println(arr1);

        System.out.println(BLACK_BOLD +"\nGet value of the 2nd row" + ANSI_RESET);
        INDArray secRow = arr1.getRow(1);
        System.out.println(secRow);

        System.out.println(BLACK_BOLD +"\nChange value of first row and second column to 0" + ANSI_RESET);
        arr1.putScalar(new int[]{0,1}, 0);
        System.out.println(arr1);

        System.out.println(BLACK_BOLD +"\nReplace 2nd row with array of ones" + ANSI_RESET);
        arr1.putRow(1, Nd4j.ones(1,5));
        System.out.println(arr1);

        System.out.println(BLACK_BOLD +"\nGet value of the 3rd row and 2nd - 4th columns" + ANSI_RESET);
        INDArray subArray = arr1.get(NDArrayIndex.point(2),NDArrayIndex.interval(1,4));
        System.out.println(subArray);

        System.out.println(BLACK_BOLD +"\nReplace the previously selected value with ones" + ANSI_RESET);
        arr1.put(
                new INDArrayIndex[]{NDArrayIndex.point(2),NDArrayIndex.interval(1,4)},
                Nd4j.ones(1,3)
        );
        System.out.println(arr1);
    }
}
