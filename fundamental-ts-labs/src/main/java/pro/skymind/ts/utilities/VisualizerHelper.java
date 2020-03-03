package pro.skymind.ts.utilities;

import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.util.Arrays;

public class VisualizerHelper {

    public static XYDataset createXYDataset(String[] seriesNames, double[][] seriesValues) {
        XYSeriesCollection dataset = new XYSeriesCollection();
        for (int i = 0; i < seriesNames.length; i++) {
            XYSeries series = new XYSeries(String.valueOf(seriesNames[i]));
            for (int j = 0; j < seriesValues[i].length; j++) {
                series.add(j, seriesValues[i][j]);
            }
            dataset.addSeries(series);
        }
        return dataset;
    }

    public static XYDataset createXYDatasetJoint(String[] seriesNames, String[] seriesNamesJoint, double[][] seriesValues, double[][] seriesValuesJoint) {
        XYSeriesCollection dataset = new XYSeriesCollection();

        // data series
        for (int i = 0; i < seriesNames.length; i++) {
            XYSeries series = new XYSeries(String.valueOf(seriesNames[i]));
            for (int j = 0; j < seriesValues[i].length; j++) {
                series.add(j, seriesValues[i][j]);
            }
            dataset.addSeries(series);
        }

        int dataSeriesLength = seriesValues[0].length;
        // forecast series
        for (int i = 0; i < seriesNamesJoint.length; i++) {
            XYSeries series = new XYSeries(String.valueOf(seriesNamesJoint[i]));
            for (int j = 0; j < seriesValuesJoint[i].length; j++) {
                series.add(j + dataSeriesLength, seriesValuesJoint[i][j]);
            }
            dataset.addSeries(series);
        }

        return dataset;
    }

    public static <T> double[] repeat(double[] arr, int newLength) {
        double[] dup = Arrays.copyOf(arr, newLength);
        for (int last = arr.length; last != 0 && last < newLength; last <<= 1) {
            System.arraycopy(dup, 0, dup, last, Math.min(last << 1, newLength) - last);
        }
        return dup;
    }
}
