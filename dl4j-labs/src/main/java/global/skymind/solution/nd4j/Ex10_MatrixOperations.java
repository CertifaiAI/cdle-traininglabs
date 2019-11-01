package global.skymind.solution.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Ex10_MatrixOperations {
    public static final String BLACK_BOLD = "\033[1;30m";
    public static final String BLUE_BOLD = "\033[1;34m";
    public static final String ANSI_RESET = "\u001B[0m";

    public static void main(String[] args) {
        INDArray array1 = Nd4j.randn(new int[]{2, 3}, 123);
        System.out.println(BLACK_BOLD + "array1" + ANSI_RESET);
        System.out.println(array1);

        INDArray array2 = Nd4j.randn(new int[]{3, 2}, 123);
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
        System.out.println();
        INDArray arr1 = Nd4j.randn(3,3);
        System.out.println(arr1);

        System.out.println();
        INDArray arr2 = Nd4j.randn(3,1);
        System.out.println(arr2);

        System.out.println();
        INDArray res = arr1.mmul(arr2);
        System.out.println(res);
    }
}
