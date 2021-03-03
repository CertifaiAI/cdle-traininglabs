/*
 * Copyright (c) 2019 Skymind Holdings Bhd.
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

package ai.certifai.training.dl1_nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

public class Ex11_BooleanIndexing {
    public static final String BLACK_BOLD = "\033[1;30m";
    public static final String BLUE_BOLD = "\033[1;34m";
    public static final String ANSI_RESET = "\u001B[0m";

    public static void main(String[] args) {
        int nRows = 3;
        int nColumns = 5;
        INDArray shape = Nd4j.create(new int[]{nRows, nColumns});
        INDArray myArray = Nd4j.randn(shape, 123);
        System.out.println(BLACK_BOLD + "Default array" + ANSI_RESET);
        System.out.println(myArray);

        //Replace negative values with zero
        BooleanIndexing.replaceWhere(myArray, 0, Conditions.lessThan(0));
        System.out.println(BLACK_BOLD + "\nReplace negative values with zero" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "BooleanIndexing.replaceWhere(myArray, 0, Conditions.lessThan(0))" + ANSI_RESET);
        System.out.println(myArray);

        //Replace values greater than one to one
        BooleanIndexing.replaceWhere(myArray, 1, Conditions.greaterThan(1));
        System.out.println(BLACK_BOLD + "\nReplace values greater than one to one" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "BooleanIndexing.replaceWhere(myArray, 1, Conditions.greaterThan(1))" + ANSI_RESET);
        System.out.println(myArray);

        //For more conditions: https://deeplearning4j.org/api/latest/org/nd4j/linalg/indexing/conditions/Condition.html

        /*
        EXERCISE:
        - Create arr1 with Nd4j.create(new float[]{1,1,1,2,2,2,3,3,3}, new int[]{3,3});
        - Set value that not equal to one to one
        */

    }
}
