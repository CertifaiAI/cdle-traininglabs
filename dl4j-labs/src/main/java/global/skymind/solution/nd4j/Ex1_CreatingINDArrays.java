package global.skymind.solution.nd4j;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JArraySizeException;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class Ex1_CreatingINDArrays {

    public static final String BLACK_BOLD = "\033[1;30m";
    public static final String BLUE_BOLD = "\033[1;34m";
    public static final String ANSI_RESET = "\u001B[0m";

    public static void main(String[] args) {

        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);

        //Initialize INDArrays with zero, one, and scalar value
        int nRows = 3;
        int nColumns = 5;
        INDArray allZeros = Nd4j.zeros(nRows, nColumns);
        System.out.println(BLUE_BOLD+"\nNd4j.zeros(nRows, nColumns)"+ANSI_RESET);
        System.out.println(allZeros);

        INDArray allOnes = Nd4j.ones(nRows, nColumns);
        System.out.println(BLUE_BOLD+"\nNd4j.ones(nRows, nColumns)"+ANSI_RESET);
        System.out.println(allOnes);

        INDArray allTens = Nd4j.valueArrayOf(nRows, nColumns, 10.0);
        System.out.println(BLUE_BOLD+"\nNd4j.valueArrayOf(nRows, nColumns, 10.0)"+ANSI_RESET);
        System.out.println(allTens);

        //Create INDArrays from double[] and double[][] (or, float/int etc Java arrays)
        double[] vectorDouble = new double[]{1,2,3};
        INDArray rowVector = Nd4j.create(vectorDouble);
        System.out.println(BLACK_BOLD + "\nCreate row vector: " + ANSI_RESET);
        System.out.println(BLUE_BOLD + "Nd4j.create(new double[]{1,2,3})" + ANSI_RESET);
        System.out.println(rowVector);

        INDArray columnVector = Nd4j.create(vectorDouble, new int[]{3,1});  //Manually specify: 3 rows, 1 column
        System.out.println(BLACK_BOLD + "\nCreate column vector: " + ANSI_RESET);
        System.out.println(BLUE_BOLD + "Nd4j.create(new double[]{1,2,3}, new int[]{3,1})" + ANSI_RESET);
        System.out.println(columnVector);

        double[][] matrixDouble = new double[][]{
                {1, 2, 3},
                {4, 5, 6}};
        INDArray matrix = Nd4j.create(matrixDouble);
        System.out.println(BLACK_BOLD + "\nCreate matrix with double[][]:" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "Nd4j.create(new double[][]{{1,2,3},{4,5,6}})" + ANSI_RESET);
        System.out.println(matrix);

        //Create random INDArrays
        //uniform random number
        INDArray uniformRand = Nd4j.rand(nRows,nColumns);
        System.out.println(BLACK_BOLD + "\nCreate uniform random array:" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "Nd4j.rand(nRows, nColumns)" + ANSI_RESET);
        System.out.println(uniformRand);

        INDArray uniformRand3D = Nd4j.rand(new int[]{2, nRows, nColumns});
        System.out.println(BLACK_BOLD + "\nCreate 3 dimensions uniform random array:" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "Nd4j.rand(new int[]{2, nRows, nColumns})" + ANSI_RESET);
        System.out.println(uniformRand3D);

        //gaussian random number
        INDArray gaussianMeanZeroUnitVariance = Nd4j.randn(new int[]{nRows, nColumns});
        System.out.println(BLACK_BOLD + "\nCreate random numbers with mean zero and standard deviation one:" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "Nd4j.randn(new int[]{nRows, nColumns})" + ANSI_RESET);
        System.out.println(gaussianMeanZeroUnitVariance);

        //Create repeatable random array using seed:
        long seed = 123;
        int[] shape = new int[]{nRows, nColumns};
        INDArray rand1 = Nd4j.rand(shape, seed);
        INDArray rand2 = Nd4j.rand(shape, seed);
        System.out.println(BLACK_BOLD +"\nUniform random arrays with same fixed seed:" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "Nd4j.rand(shape, seed)" + ANSI_RESET);
        System.out.println(BLACK_BOLD + "rand1" + ANSI_RESET);
        System.out.println(rand1);
        System.out.println(BLACK_BOLD + "rand2" + ANSI_RESET);
        System.out.println(rand2);

        //Other miscellaneous methods
        INDArray identityMatrix = Nd4j.eye(3);
        System.out.println(BLACK_BOLD +"\nCreate identity matrix:" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "Nd4j.eye(3)" + ANSI_RESET);
        System.out.println(identityMatrix);

        INDArray linspace = Nd4j.linspace(1,20,10); //Values 1 to 20, in 10 steps
        System.out.println(BLACK_BOLD +"\nGenerate a vector with range of values (start from 1 to 20 in 10 steps):" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "Nd4j.linspace(1,20,10)" + ANSI_RESET);
        System.out.println(linspace);

        INDArray diagMatrix = Nd4j.diag(Nd4j.create(new double[]{4,5,6}));
        System.out.println(BLACK_BOLD +"\nCreate square matrix, with vector along the diagonal:" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "Nd4j.diag(Nd4j.create(new double[]{4,5,6}))" + ANSI_RESET);
        System.out.println(diagMatrix);

        /*
        EXERCISE:
        - Create array with shape (3,2) initialize it with 0
        - Create array with shape (5,5) initialize it with 5
        - Create the following array
            | 0  0  0 |
            | 1  1  1 |
            | 2  2  2 |
        - Create array with shape (3,3,3) initialize it with gaussian random number
        - Create a vector with range of 1 - 100 in 20 steps
        */
        INDArray ex1 = Nd4j.zeros(3, 2);
        System.out.println(BLACK_BOLD +"\nCreate array with shape (3,2) initialize it with 0" + ANSI_RESET);
        System.out.println(ex1);

        INDArray ex2 = Nd4j.valueArrayOf(5, 5,5);
        System.out.println(BLACK_BOLD +"\nCreate array with shape (5,5) initialize it with 5" + ANSI_RESET);
        System.out.println(ex2);

        INDArray ex3 = Nd4j.create(new double[]{0,0,0,1,1,1,2,2,2}, new int[]{3,3});
        System.out.println(BLACK_BOLD +"\nCreate the following array" + ANSI_RESET);
        System.out.println(ex3);

        INDArray ex4 = Nd4j.randn(new int[]{3,3,3});
        System.out.println(BLACK_BOLD +"\nCreate array with shape (3,3,3) initialize it with gaussian random number" + ANSI_RESET);
        System.out.println(ex4);

        INDArray ex5 = Nd4j.linspace(1,100,20);
        System.out.println(BLACK_BOLD +"\nCreate a vector with range of 1 to 100 in 20 steps" + ANSI_RESET);
        System.out.println(ex5);

    }
}
