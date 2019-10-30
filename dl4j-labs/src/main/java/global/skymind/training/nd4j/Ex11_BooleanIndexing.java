package global.skymind.training.nd4j;

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
        INDArray myArray = Nd4j.randn(123, new long[]{nRows, nColumns});
        System.out.println(BLACK_BOLD + "Default array" + ANSI_RESET);
        System.out.println(myArray);

        //Replace negative values with zero
//        BooleanIndexing.replaceWhere(myArray, 0, Conditions.lessThan(0));
//        System.out.println(BLACK_BOLD + "\nReplace negative values with zero" + ANSI_RESET);
//        System.out.println(BLUE_BOLD + "BooleanIndexing.replaceWhere(myArray, 0, Conditions.lessThan(0))" + ANSI_RESET);
//        System.out.println(myArray);

        //Replace values greater than one to one
//        BooleanIndexing.replaceWhere(myArray, 1, Conditions.greaterThan(1));
//        System.out.println(BLACK_BOLD + "\nReplace values greater than one to one" + ANSI_RESET);
//        System.out.println(BLUE_BOLD + "BooleanIndexing.replaceWhere(myArray, 1, Conditions.greaterThan(1))" + ANSI_RESET);
//        System.out.println(myArray);

    }
}
