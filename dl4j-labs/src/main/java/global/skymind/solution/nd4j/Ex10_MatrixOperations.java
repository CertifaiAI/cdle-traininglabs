package global.skymind.solution.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Ex10_MatrixOperations {
    public static final String BLACK_BOLD = "\033[1;30m";
    public static final String BLUE_BOLD = "\033[1;34m";
    public static final String ANSI_RESET = "\u001B[0m";

    public static void main(String[] args) {
        INDArray shape1 = Nd4j.create(new int[]{2, 3});
        INDArray array1 = Nd4j.randn(shape1, 123);
        System.out.println(BLUE_BOLD + "array1"+ ANSI_RESET);
        System.out.println(array1);

        INDArray shape2 = Nd4j.create(new int[]{3, 2});
        INDArray array2 = Nd4j.randn(shape2, 123);
        System.out.println(BLUE_BOLD + "array2" + ANSI_RESET);
        System.out.println(array2);

        // Matrix multiplication
        INDArray mmulArray = array1.mmul(array2);
        System.out.println(BLUE_BOLD + "\nMatrix multiplication of array1 and array2" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.addRowVector(rowVector)" + ANSI_RESET);
        System.out.println(mmulArray);

        /*
        EXERCISE:
        - Create arr1 with shape(3,3) initialize with random value
        - Create arr2 with shape(3,1) initialize with random value
        - Perform matrix multiplication of arr1 and arr2
        */
        System.out.println(BLUE_BOLD +"\nCreate arr1 with shape(3,3) initialize with random value" + ANSI_RESET);
        INDArray arr1 = Nd4j.randn(3,3);
        System.out.println(arr1);

        System.out.println(BLUE_BOLD +"\nCreate arr2 with shape(3,1) initialize with random value" + ANSI_RESET);
        INDArray arr2 = Nd4j.randn(3,1);
        System.out.println(arr2);

        System.out.println(BLUE_BOLD +"\nPerform matrix multiplication of arr1 and arr2" + ANSI_RESET);
        INDArray res = arr1.mmul(arr2);
        System.out.println(res);
    }
}
