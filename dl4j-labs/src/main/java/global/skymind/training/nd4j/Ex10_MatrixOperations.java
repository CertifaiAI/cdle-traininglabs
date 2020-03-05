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

public class Ex10_MatrixOperations {
    public static final String BLACK_BOLD = "\033[1;30m";
    public static final String BLUE_BOLD = "\033[1;34m";
    public static final String ANSI_RESET = "\u001B[0m";

    public static void main(String[] args) {
        INDArray shape1 = Nd4j.create(new int[]{2, 3});
        INDArray array1 = Nd4j.randn(shape1, 123);
        System.out.println(BLACK_BOLD + "array1" + ANSI_RESET);
        System.out.println(array1);

        INDArray shape2 = Nd4j.create(new int[]{3, 2});
        INDArray array2 = Nd4j.randn(shape2, 123);
        System.out.println(BLACK_BOLD + "array2" + ANSI_RESET);
        System.out.println(array2);

        // Matrix multiplication
        INDArray mmulArray = array1.mmul(array2);
        System.out.println(BLACK_BOLD + "\nMatrix multiplication of array1 and array2" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.addRowVector(rowVector)" + ANSI_RESET);
        System.out.println(mmulArray);

        /*
        EXERCISE:
        - Create arr1 with shape(3,3) initialize with random value
        - Create arr2 with shape(3,1) initialize with random value
        - Perform matrix multiplication of arr1 and arr2
        */

    }
}
