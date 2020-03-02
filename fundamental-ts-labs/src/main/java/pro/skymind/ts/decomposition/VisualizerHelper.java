package pro.skymind.ts.decomposition;

import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

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

    public static XYDataset createXYDatasetJoint(String[] seriesNames, String[] seriesNamesJoint, double[][] seriesValues, double[][] seriesValuesJoint, int length) {
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
}
