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

package ai.certifai.solution.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Ex7_ReductionOperations {
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

        //First, let's consider whole array reductions (the results are in double precision):
        double minValue = myArray.minNumber().doubleValue();
        double maxValue = myArray.maxNumber().doubleValue();
        double sum = myArray.sumNumber().doubleValue();
        double avg = myArray.meanNumber().doubleValue();
        double stdev = myArray.stdNumber().doubleValue();
        System.out.println(BLACK_BOLD + "\nWhole array reductions (min, max, sum, mean, std)" + ANSI_RESET);
        System.out.println("minValue:       " + minValue);
        System.out.println("maxValue:       " + maxValue);
        System.out.println("sum:            " + sum);
        System.out.println("average:        " + avg);
        System.out.println("standard dev.:  " + stdev);

        //Alternative to get sum of the array
        INDArray sumArray2 = myArray.sum();
        System.out.println(BLACK_BOLD + "\nSum of the array" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.sum()" + ANSI_RESET);
        System.out.println(sumArray2);

        //Sum array along dimension 0 (vertical)
        INDArray sumArrayVer = myArray.sum(0);
        System.out.println(BLACK_BOLD + "\nSum array along dimension 0 (vertical)" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.sum(0)" + ANSI_RESET);
        System.out.println(sumArrayVer);

        //Sum array along dimension 1 (horizontal)
        INDArray sumArrayHor = myArray.sum(1);
        System.out.println(BLACK_BOLD + "\nSum array along dimension 1 (horizontal)" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.sum(1)" + ANSI_RESET);
        System.out.println(sumArrayHor);

        //Average of the array
        INDArray meanArray = myArray.mean();
        System.out.println(BLACK_BOLD + "\nAverage of the array" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.mean()" + ANSI_RESET);
        System.out.println(meanArray);

        /*
        EXERCISE:
        - Create arr1 with shape(3,3) initialize with random value
        - Get min value from the arr1 along dimension 0
        - Get max value from the arr1
        */
        System.out.println(BLACK_BOLD +"\nCreate arr1 with shape(3,3) initialize with random value" + ANSI_RESET);
        INDArray arr1 = Nd4j.randn(3,3);
        System.out.println(arr1);

        System.out.println(BLACK_BOLD +"\nGet min value from the arr1 along dimension 0" + ANSI_RESET);
        INDArray min = arr1.min(0);
        System.out.println(min);

        System.out.println(BLACK_BOLD +"\nGet max value from the arr1" + ANSI_RESET);
        INDArray max = arr1.max();
        System.out.println(max);
    }
}
