package global.skymind.solution.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMin;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Ex8_IndexReductionOperations {
    public static final String BLACK_BOLD = "\033[1;30m";
    public static final String BLUE_BOLD = "\033[1;34m";
    public static final String ANSI_RESET = "\u001B[0m";

    public static void main(String[] args) {
        int nRows = 3;
        int nColumns = 5;
        INDArray myArray = Nd4j.rand(nRows, nColumns, 123);
        System.out.println(BLACK_BOLD + "Default array" + ANSI_RESET);
        System.out.println(myArray);

        //Get the index of maximum value
        INDArray maxIndex = myArray.argMax();
        System.out.println(BLACK_BOLD + "\nGet the index of maximum value" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.argMax()" + ANSI_RESET);
        System.out.println(maxIndex);

        //Get the index of maximum value in vertical direction or dimension 0
        INDArray maxIndexVer = myArray.argMax(0);
        System.out.println(BLACK_BOLD + "\nGet the index of maximum value in vertical direction or dimension 0" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.argMax(0)" + ANSI_RESET);
        System.out.println(maxIndexVer);

        //Get the index of maximum value in horizontal direction or dimension 1
        INDArray maxIndexHor = myArray.argMax(1);
        System.out.println(BLACK_BOLD + "\nGet the index of maximum value in horizontal direction or dimension 1" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.argMax(1)" + ANSI_RESET);
        System.out.println(maxIndexHor);

        //Index of the min value, along dimension 0
        INDArray minIndexAlongDim0 = Nd4j.getExecutioner().exec(new IMin(myArray, 0));
        System.out.println(BLACK_BOLD + "\nGet the index of minimum value along dimension 0:" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "Nd4j.getExecutioner().exec(new IMin(myArray, 0))" + ANSI_RESET);
        System.out.println(minIndexAlongDim0);

        /*
        EXERCISE:
        - Create arr1 with shape(3,3) initialize with random value
        - Get index of max value from the arr1 along dimension 1
        */
        System.out.println(BLACK_BOLD +"\nCreate arr1 with shape(3,3) initialize with random value" + ANSI_RESET);
        INDArray arr1 = Nd4j.randn(3,3);
        System.out.println(arr1);

        System.out.println(BLACK_BOLD +"\nGet index of max value from the arr1 along dimension 1" + ANSI_RESET);
        INDArray maxIdx = arr1.argMax(1);
        System.out.println(maxIdx);
    }
}
