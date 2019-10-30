package global.skymind.training.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class Ex4_GettingandSettingINDArrays {
    public static final String BLACK_BOLD = "\033[1;30m";
    public static final String BLUE_BOLD = "\033[1;34m";
    public static final String ANSI_RESET = "\u001B[0m";

    public static void main(String[] args) {
        int nRows = 3;
        int nColumns = 5;
        INDArray myArray = Nd4j.rand(123, new long[]{nRows,nColumns});
        System.out.println(BLACK_BOLD + "Default array" + ANSI_RESET);
        System.out.println(myArray);

        //Get scalar value
//        INDArray scalarVal = myArray.getScalar(0,0); //try getDouble, getInt, and getFloat to see the different
//        System.out.println(BLACK_BOLD + "\nGet scalar value, position row 1 and column 1" + ANSI_RESET);
//        System.out.println(BLUE_BOLD + "myArray.getScalar(0,0)" + ANSI_RESET);
//        System.out.println(scalarVal);

        //Set scalar value
//        myArray.putScalar(6,0);
//        System.out.println(BLACK_BOLD + "\nSet scalar value by index, set value of index 6 to 0" + ANSI_RESET);
//        System.out.println(BLUE_BOLD + "myArray.getScalar(6,0)" + ANSI_RESET);
//        System.out.println(myArray);

//        myArray.putScalar(new int[]{1,4},10);
//        System.out.println(BLACK_BOLD + "\nSet scalar value based on given location, row 2 and column 5" + ANSI_RESET);
//        System.out.println(BLUE_BOLD + "myArray.putScalar(new int[]{1,4},10)" + ANSI_RESET);
//        System.out.println(myArray);

        //Get row or column
//        INDArray row = myArray.getRow(0);
//        System.out.println(BLACK_BOLD + "\nGet the first row from myArray" + ANSI_RESET);
//        System.out.println(BLUE_BOLD + "myArray.getRow(0)" + ANSI_RESET);
//        System.out.println(row);

//        INDArray rows = myArray.getRows(0,2);
//        System.out.println(BLACK_BOLD + "\nGet multiple rows from myArray" + ANSI_RESET);
//        System.out.println(BLUE_BOLD + "myArray.getRows(0,2)" + ANSI_RESET);
//        System.out.println(rows);

        //Set row or column
//        myArray.putRow(2, Nd4j.zeros(5));
//        System.out.println(BLACK_BOLD + "\nSet row number 2 to zero" + ANSI_RESET);
//        System.out.println(BLUE_BOLD + "myArray.putRow(2, Nd4j.zeros(5))" + ANSI_RESET);
//        System.out.println(myArray);

        //Get an arbitrary sub-arrays based on certain indexes
//        INDArray first3Columns = myArray.get(NDArrayIndex.all(), NDArrayIndex.interval(0,3));
//        System.out.println(BLACK_BOLD + "\nGet first 3 columns" + ANSI_RESET);
//        System.out.println(BLUE_BOLD + "myArray.get(NDArrayIndex.all(), NDArrayIndex.interval(0,3))" + ANSI_RESET);
//        System.out.println(first3Columns);

//        INDArray secondRow = myArray.get(NDArrayIndex.point(1), NDArrayIndex.all());
//        System.out.println(BLACK_BOLD + "\nGet second row" + ANSI_RESET);
//        System.out.println(BLUE_BOLD + "myArray.get(NDArrayIndex.point(1), NDArrayIndex.all())" + ANSI_RESET);
//        System.out.println(secondRow);

//        INDArray oddColumns = myArray.get(NDArrayIndex.all(), NDArrayIndex.interval(0,2,5));
//        System.out.println(BLACK_BOLD + "\nGet all rows and odd column" + ANSI_RESET);
//        System.out.println(BLUE_BOLD + "myArray.get(NDArrayIndex.all(), NDArrayIndex.interval(0,2,5))" + ANSI_RESET);
//        System.out.println(oddColumns);

        //Set an arbitrary sub-arrays based on certain indexes
//        myArray.put(
//                new INDArrayIndex[]{NDArrayIndex.point(0),NDArrayIndex.all()},
//                Nd4j.zeros(1,5)
//        );
//        System.out.println(BLACK_BOLD + "\nSet a first row and all columns to zero" + ANSI_RESET);
//        System.out.println(BLUE_BOLD + "myArray.put(new INDArrayIndex[]{NDArrayIndex.point(0),NDArrayIndex.all()},Nd4j.zeros(1,5));" + ANSI_RESET);
//        System.out.println(myArray);

        //Alternative way
//        myArray.put(
//                myArray.get(NDArrayIndex.point(0), NDArrayIndex.all()),
//                Nd4j.zeros(1,5)
//        );
//        System.out.println(BLACK_BOLD + "\nSet a first row and all columns to zero" + ANSI_RESET);
//        System.out.println(BLUE_BOLD + "myArray.put(myArray.get(NDArrayIndex.point(0), NDArrayIndex.all()),Nd4j.zeros(1,5));" + ANSI_RESET);
//        System.out.println(myArray);
    }
}
