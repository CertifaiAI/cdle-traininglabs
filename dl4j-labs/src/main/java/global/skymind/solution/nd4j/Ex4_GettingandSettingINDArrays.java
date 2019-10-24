package global.skymind.solution.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class Ex4_GettingandSettingINDArrays {
    public static final String BLACK_BOLD = "\033[1;30m";  // BLACK
    public static final String RED_BOLD = "\033[1;31m";    // RED
    public static final String GREEN_BOLD = "\033[1;32m";  // GREEN
    public static final String YELLOW_BOLD = "\033[1;33m"; // YELLOW
    public static final String BLUE_BOLD = "\033[1;34m";   // BLUE
    public static final String PURPLE_BOLD = "\033[1;35m"; // PURPLE
    public static final String CYAN_BOLD = "\033[1;36m";   // CYAN
    public static final String WHITE_BOLD = "\033[1;37m";  // WHITE
    public static final String ANSI_RESET = "\u001B[0m";

    public static void main(String[] args) {
        int nRows = 3;
        int nColumns = 5;
        INDArray myArray = Nd4j.rand(new int[]{nRows,nColumns}, 123);
        System.out.println(BLACK_BOLD + "Default array" + ANSI_RESET);
        System.out.println(myArray);

        //Get scalar value
        INDArray scalarVal = myArray.getScalar(0,0); //try getDouble, getInt, and getFloat to see the different
        System.out.println(BLACK_BOLD + "\nGet scalar value" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.getScalar(0,0)" + ANSI_RESET);
        System.out.println(scalarVal);

        //Set scalar value
        myArray.putScalar(6,0);
        System.out.println(BLACK_BOLD + "\nSet scalar value by index" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.getScalar(6,0)" + ANSI_RESET);
        System.out.println(myArray);

        myArray.putScalar(new int[]{1,4},10);
        System.out.println(BLACK_BOLD + "\nSet scalar value based on given location" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.putScalar(new int[]{1,4},10)" + ANSI_RESET);
        System.out.println(myArray);

        //Get row or column
        INDArray row = myArray.getRow(0);
        System.out.println(BLACK_BOLD + "\nGet the first row from myArray" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.getRow(0)" + ANSI_RESET);
        System.out.println(row);

        INDArray rows = myArray.getRows(0,2);
        System.out.println(BLACK_BOLD + "\nGet multiple rows from myArray" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.getRows(0,2)" + ANSI_RESET);
        System.out.println(rows);

        //Set row or column
        myArray.putRow(2, Nd4j.zeros(5));
        System.out.println(BLACK_BOLD + "\nSet row number 2 to zero" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.putRow(2, Nd4j.zeros(5))" + ANSI_RESET);
        System.out.println(myArray);

        //Get an arbitrary sub-arrays based on certain indexes
        INDArray first3Columns = myArray.get(NDArrayIndex.all(), NDArrayIndex.interval(0,3));
        System.out.println(BLACK_BOLD + "\nGet first 3 columns" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.get(NDArrayIndex.all(), NDArrayIndex.interval(0,3))" + ANSI_RESET);
        System.out.println(first3Columns);

        INDArray secondRow = myArray.get(NDArrayIndex.point(1), NDArrayIndex.all());
        System.out.println(BLACK_BOLD + "\nGet second row" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.get(NDArrayIndex.point(1), NDArrayIndex.all())" + ANSI_RESET);
        System.out.println(secondRow);

        INDArray oddColumns = myArray.get(NDArrayIndex.all(), NDArrayIndex.interval(0,2,5));
        System.out.println(BLACK_BOLD + "\nGet all rows and odd column" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.get(NDArrayIndex.all(), NDArrayIndex.interval(0,2,5))" + ANSI_RESET);
        System.out.println(oddColumns);

        //Set an arbitrary sub-arrays based on certain indexes
        myArray.put(
                new INDArrayIndex[]{NDArrayIndex.point(0),NDArrayIndex.all()},
                Nd4j.zeros(1,5)
        );
        System.out.println(BLACK_BOLD + "\nSet a first row and all columns to zero" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.put(new INDArrayIndex[]{NDArrayIndex.point(0),NDArrayIndex.all()},Nd4j.zeros(1,5));" + ANSI_RESET);
        System.out.println(myArray);

        //Alternative way
        myArray.put(
                myArray.get(NDArrayIndex.point(0), NDArrayIndex.all()),
                Nd4j.zeros(1,5)
        );
        System.out.println(BLACK_BOLD + "\nSet a first row and all columns to zero" + ANSI_RESET);
        System.out.println(BLUE_BOLD + "myArray.put(myArray.get(NDArrayIndex.point(0), NDArrayIndex.all()),Nd4j.zeros(1,5));" + ANSI_RESET);
        System.out.println(myArray);
    }
}
