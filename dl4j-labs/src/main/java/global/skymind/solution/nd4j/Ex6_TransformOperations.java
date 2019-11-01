package global.skymind.solution.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Ex6_TransformOperations {
    public static final String BLACK_BOLD = "\033[1;30m";
    public static final String BLUE_BOLD = "\033[1;34m";
    public static final String ANSI_RESET = "\u001B[0m";

    public static void main(String[] args) {
        int nRows = 3;
        int nColumns = 5;
        INDArray myArray = Nd4j.rand(new int[]{nRows, nColumns}, 123);
        System.out.println(BLACK_BOLD + "Default array" + ANSI_RESET);
        System.out.println(myArray);

        //array log
        INDArray logArray = Transforms.log(myArray);
        System.out.println(BLACK_BOLD + "\nArray log transform" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "Transforms.log(myArray)" + ANSI_RESET);
        System.out.println(logArray);

        //array absolute value
        INDArray absArray = Transforms.abs(logArray);
        System.out.println(BLACK_BOLD + "\nArray absolute transform" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "Transforms.abs(myArray)" + ANSI_RESET);
        System.out.println(absArray);

        //Round up array
        INDArray roundUpArray = Transforms.ceil(absArray);
        System.out.println(BLACK_BOLD + "\nRound up array" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "Transforms.ceil(absArray)" + ANSI_RESET);
        System.out.println(roundUpArray);

        //Array sigmoid function
        INDArray sigmoidArray = Transforms.sigmoid(myArray);
        System.out.println(BLACK_BOLD + "\nArray sigmoid function" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "Transforms.sigmoid(myArray)" + ANSI_RESET);
        System.out.println(sigmoidArray);

        // For more operation: https://deeplearning4j.org/api/latest/org/nd4j/linalg/api/ops/TransformOp.html

        /*
        EXERCISE:
        - Create arr1 with shape(3,3) initialize with random value
        - Perform TanH operation on arr1
        - Perform round operation on arr1
        */
        System.out.println(BLACK_BOLD +"\nCreate arr1 with shape(3,3) initialize with random value" + ANSI_RESET);
        INDArray arr1 = Nd4j.randn(3,3);
        System.out.println(arr1);

        System.out.println(BLACK_BOLD +"\nPerform TanH operation on arr1" + ANSI_RESET);
        INDArray arrTanh = Transforms.tanh(arr1);
        System.out.println(arrTanh);

        System.out.println(BLACK_BOLD +"\nPerform round operation on arr1" + ANSI_RESET);
        INDArray arrRound = Transforms.round(arr1);
        System.out.println(arrRound);
    }
}
