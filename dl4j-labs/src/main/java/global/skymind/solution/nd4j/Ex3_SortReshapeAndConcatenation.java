package global.skymind.solution.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class Ex3_SortReshapeAndConcatenation {
    public static final String BLACK_BOLD = "\033[1;30m";
    public static final String BLUE_BOLD = "\033[1;34m";
    public static final String ANSI_RESET = "\u001B[0m";

    public static void main(String[] args) {
        int nRows = 3;
        int nColumns = 5;
        INDArray myArray = Nd4j.rand(nRows, nColumns, 123);
        System.out.println(BLACK_BOLD + "Default array" + ANSI_RESET);
        System.out.println(myArray);

        //Sorting
        INDArray sortedArrayVertical = Nd4j.sort(myArray.dup(),0,true);
        System.out.println(BLACK_BOLD + "\nSort at dimension 0 and ascending" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "Nd4j.sort(myArray,0,true)" + ANSI_RESET);
        System.out.println(sortedArrayVertical);

        INDArray sortedArrayHorizontal = Nd4j.sort(myArray.dup(),1,false);
        System.out.println(BLACK_BOLD + "\nSort at dimension 1 and descending" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "Nd4j.sort(myArray,1,false)" + ANSI_RESET);
        System.out.println(sortedArrayHorizontal);

        //Flatten
        INDArray flattened = Nd4j.toFlattened(myArray);
        System.out.println(BLACK_BOLD + "\nFlatten array" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "Nd4j.toFlattened(myArray)" + ANSI_RESET);
        System.out.println(flattened);

        //Transpose
        INDArray transposed = myArray.transpose();
        System.out.println(BLACK_BOLD + "\nTranspose array" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.transpose()" + ANSI_RESET);
        System.out.println(transposed);

        //Reshape
        INDArray reshaped = myArray.reshape(5,3);
        System.out.println(BLACK_BOLD + "\nReshaped array" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.reshape(5,3)" + ANSI_RESET);
        System.out.println(reshaped);

        //Concatenate via stacking
        INDArray newArray = Nd4j.zeros(nRows, nColumns);
        System.out.println(BLACK_BOLD + "\nCreate newArray" + ANSI_RESET);
        System.out.println(newArray);

        INDArray vStacked = Nd4j.vstack(myArray, newArray);
        System.out.println(BLACK_BOLD + "\nStack myArray and newArray vertically" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "Nd4j.vstack(myArray, newArray)" + ANSI_RESET);
        System.out.println(vStacked);

        INDArray hStacked = Nd4j.hstack(myArray, newArray);
        System.out.println(BLACK_BOLD + "\nStack myArray and newArray horizontally" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "Nd4j.hstack(myArray, newArray)" + ANSI_RESET);
        System.out.println(hStacked);

        //Concatenation, combines arrays along a dimension
        int[] shape = new int[]{2,3,2};
        INDArray array1 = Nd4j.zeros(shape);
        INDArray array2 = Nd4j.ones(shape);

        INDArray concatenated = Nd4j.concat(0, array1, array2);
        System.out.println(BLACK_BOLD + "\nconcatenate array1 and array2 in dimension 0" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "Nd4j.concat(0, array1, array2)" + ANSI_RESET);
        System.out.println("Shape: " + Arrays.toString(concatenated.shape()));
        System.out.println(concatenated);

        INDArray concatenated2 = Nd4j.concat(1, array1, array2);
        System.out.println(BLACK_BOLD + "\nconcatenate array1 and array2 in dimension 1" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "Nd4j.concat(1, array1, array2)" + ANSI_RESET);
        System.out.println("Shape: " + Arrays.toString(concatenated2.shape()));
        System.out.println(concatenated2);

        /*
        EXERCISE:
        - Create arr1 with shape(2,3) initialize with random value
        - Create arr2 with shape(5,2) initialize with random value
        - Concatenate array2 to array1 (use reshape, flatten, or transpose if needed)
        - Sort(descending) the concatenated array in dimension 1
        */
        System.out.println(BLACK_BOLD +"\nCreate arr1 with shape(2,3) initialize with random value" + ANSI_RESET);
        INDArray arr1 = Nd4j.randn(2,3);
        System.out.println(arr1);

        System.out.println(BLACK_BOLD +"\nCreate arr2 with shape(5,2) initialize with random value" + ANSI_RESET);
        INDArray arr2 = Nd4j.randn(5,2);
        System.out.println(arr2);

        System.out.println(BLACK_BOLD +"\nConcatenate array2 to array1 (use reshape, flatten, or transpose if needed)" + ANSI_RESET);
        INDArray concated = Nd4j.concat(1, arr1, arr2.transpose());
        System.out.println(concated);

        System.out.println(BLACK_BOLD +"\nSort(descending) the concatenated array in dimension 1" + ANSI_RESET);
        INDArray concatedSorted = Nd4j.sort(concated,1,false);
        System.out.println(concatedSorted);

    }
}
