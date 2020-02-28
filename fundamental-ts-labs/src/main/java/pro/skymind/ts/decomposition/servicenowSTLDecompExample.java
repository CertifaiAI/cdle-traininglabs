package pro.skymind.ts.decomposition;

import com.github.servicenow.ds.stats.stl.SeasonalTrendLoess;
import com.github.signaflo.timeseries.TestData;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.util.Arrays;

public class servicenowSTLDecompExample extends JFrame {
    public servicenowSTLDecompExample(String title) {
        double[] values = TestData.debitcards.asArray(); // Monthly time-series data

        SeasonalTrendLoess.Builder builder = new SeasonalTrendLoess.Builder();
        SeasonalTrendLoess smoother = builder.
                setPeriodLength(12).    // Data has a period of 12
                setSeasonalWidth(35).   // Monthly data smoothed over 35 years
                setNonRobust().         // Not expecting outliers, so no robustness iterations
                buildSmoother(values);

        SeasonalTrendLoess.Decomposition stl = smoother.decompose();
        double[] seasonal = stl.getSeasonal();
        double[] trend = stl.getTrend();
        double[] residual = stl.getResidual();

        Arrays.stream(seasonal).forEach(num -> System.out.print(num + ","));
        System.out.println();
        Arrays.stream(trend).forEach(num -> System.out.print(num + ","));
        System.out.println();
        Arrays.stream(residual).forEach(num -> System.out.print(num + ","));

        JFreeChart chart = ChartFactory.createXYLineChart(
                "STL Decomposition", // Chart title
                "Month", // X-Axis Label
                "Debit card usages", // Y-Axis Label
                createDataset(values, seasonal, trend, residual),
                PlotOrientation.VERTICAL,
                true,
                true,
                true
        );
        ChartPanel panel = new ChartPanel(chart);

        setContentPane(panel);
    }

    private XYDataset createDataset(double[] values, double[] seasonal, double[] trend, double[] residual) {
        String[] seriesArr = new String[]{
                "values",
                "seasonal",
                "trend",
                "residual"
        };
        double[][] valueArr = new double[][]{values, seasonal, trend, residual};

        XYSeriesCollection dataset = new XYSeriesCollection();
        for (int i = 0; i < seriesArr.length; i++) {
            XYSeries series = new XYSeries(String.valueOf(seriesArr[i]));
            for (int j = 0; j < valueArr[i].length; j++) {
                series.add( j, valueArr[i][j]);
            }
            dataset.addSeries(series);
        }
        return dataset;
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            servicenowSTLDecompExample example = new servicenowSTLDecompExample("Line Chart Example");
            example.setAlwaysOnTop(true);
            example.pack();
            example.setSize(1200, 800);
            example.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
            example.setVisible(true);
        });
    }
}
