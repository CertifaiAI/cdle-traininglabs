package global.skymind.solution.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Ex10_MatrixOperations {
    public static final String BLACK_BOLD = "\033[1;30m";
    public static final String BLUE_BOLD = "\033[1;34m";
    public static final String ANSI_RESET = "\u001B[0m";

    public static void main(String[] args) {
        INDArray array1 = Nd4j.randn(123, new long[]{2, 3});
        System.out.println(BLACK_BOLD + "array1" + ANSI_RESET);
        System.out.println(array1);

        INDArray array2 = Nd4j.randn(123, new long[]{3, 2});
        System.out.println(BLACK_BOLD + "array2" + ANSI_RESET);
        System.out.println(array2);

        // Matrix multiplication
        INDArray mmulArray = array1.mmul(array2);
        System.out.println(BLACK_BOLD + "\nMatrix multiplication of array1 and array2" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.addRowVector(rowVector)" + ANSI_RESET);
        System.out.println(mmulArray);
    }
}
