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

public class servicenowSTLDecompExample extends JFrame {

    public servicenowSTLDecompExample(String title) {
        double[] values = TestData.debitcards.asArray(); // Monthly time-series data

        SeasonalTrendLoess smoother = new SeasonalTrendLoess.Builder().
                setPeriodLength(12).    // Data has a period of 12
                setSeasonalWidth(156).   // Monthly data smoothed over 35 years
                setNonRobust().         // Not expecting outliers, so no robustness iterations
                buildSmoother(values);

        SeasonalTrendLoess.Decomposition stl = smoother.decompose();

        double[] seasonal = stl.getSeasonal();
        double[] trend = stl.getTrend();
        double[] residual = stl.getResidual();

        XYDataset dataset = VisualizerHelper.createXYDataset(
                new String[]{"values", "seasonal", "trend", "residual"},
                new double[][]{values, seasonal, trend, residual}
        );

        JFreeChart chart = ChartFactory.createXYLineChart(
                title, // Chart title
                "Month", // X-Axis Label
                "Debit card usages", // Y-Axis Label
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                true
        );
        ChartPanel panel = new ChartPanel(chart);
        setTitle(title);
        setContentPane(panel);
    }


    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            servicenowSTLDecompExample example = new servicenowSTLDecompExample("STL Decomposition");
            example.setAlwaysOnTop(true);
            example.pack();
            example.setSize(1200, 800);
            example.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
            example.setVisible(true);
        });
    }
}
