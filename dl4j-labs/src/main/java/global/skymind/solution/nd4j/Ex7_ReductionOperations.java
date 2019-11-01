package global.skymind.solution.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Ex7_ReductionOperations {
    public static final String BLACK_BOLD = "\033[1;30m";
    public static final String BLUE_BOLD = "\033[1;34m";
    public static final String ANSI_RESET = "\u001B[0m";

    public static void main(String[] args) {
        int nRows = 3;
        int nColumns = 5;
        INDArray myArray = Nd4j.randn(new int[]{nRows, nColumns}, 123);
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
        System.out.println();
        INDArray arr1 = Nd4j.randn(3,3);
        System.out.println(arr1);

        System.out.println();
        INDArray min = arr1.min(0);
        System.out.println(min);

        System.out.println();
        INDArray max = arr1.max();
        System.out.println(max);
    }
}
