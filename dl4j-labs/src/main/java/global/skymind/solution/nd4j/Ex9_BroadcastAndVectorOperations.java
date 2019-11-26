package global.skymind.solution.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Ex9_BroadcastAndVectorOperations {
    public static final String BLACK_BOLD = "\033[1;30m";
    public static final String BLUE_BOLD = "\033[1;34m";
    public static final String ANSI_RESET = "\u001B[0m";

    public static void main(String[] args) {
        int nRows = 3;
        int nColumns = 5;
        INDArray myArray = Nd4j.randn(nRows, nColumns, 123);
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
        System.out.println(BLACK_BOLD +"\nCreate arr1 with shape(3,3) initialize with random value" + ANSI_RESET);
        INDArray arr1 = Nd4j.randn(3,3);
        System.out.println(arr1);

        System.out.println(BLACK_BOLD +"\nDivide arr1 with row vector of [2,2,2]" + ANSI_RESET);
        INDArray div = arr1.divRowVector(Nd4j.valueArrayOf(new int[]{1,3}, 2));
        System.out.println(div);

        System.out.println(BLACK_BOLD +"\nMultiply arr1 with column vector of [1,2,3]" + ANSI_RESET);
        INDArray mul = arr1.mulColumnVector(Nd4j.create(new float[]{1,2,3}, 3, 1));
        System.out.println(mul);
    }
}
