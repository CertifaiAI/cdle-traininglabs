package pro.skymind.ts.arima;

import com.github.signaflo.math.stats.distributions.Normal;
import com.github.signaflo.timeseries.TimePeriod;
import com.github.signaflo.timeseries.TimeSeries;
import com.github.signaflo.timeseries.forecast.Forecast;
import com.github.signaflo.timeseries.model.arima.Arima;
import com.github.signaflo.timeseries.model.arima.ArimaOrder;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYDataset;
import pro.skymind.ts.utilities.VisualizerHelper;

import javax.swing.*;

public class signafloARIMASeasonalExample extends JFrame {

    public signafloARIMASeasonalExample(String title) {
        // First, we'll fill in 15 weeks worth of daily data with an extremely simple
        // simulated data generating process.
        Normal normal = new Normal(); // Create a normal distribution with mean 0 an sd of 1.

        double[] values = new double[105];
        for (int i = 0; i < values.length; i++) {
            values[i] = normal.rand();
        }

        // Assumes Monday corresponds to 0.
        for (int fri = 4; fri < values.length; fri += 7) {
            values[fri] += 1.0;
            values[fri + 1] += 2.0;
            values[fri + 2] -= 1.0;
        }

        // Second, we'll create a daily time series from those values.
        TimePeriod day = TimePeriod.oneDay();

        TimeSeries series = TimeSeries.from(day, values);

        // Third, we'll create an ArimaOrder with a seasonal component.
        ArimaOrder order = ArimaOrder.order(0, 0, 0, 1, 1, 1);

        // Fourth, we create an ARIMA model with the series, the order,
        // and the weekly seasonality.
        TimePeriod week = TimePeriod.oneWeek();

        Arima model = Arima.model(series, order, week);

        // Finally, generate a forecast for next week using the model
        Forecast forecast = model.forecast(7);
        System.out.println(forecast);

        // visualization
        XYDataset dataset = VisualizerHelper.createXYDatasetJoint(
                new String[]{"series"},
                new String[]{"forecast", "lower", "upper"},
                new double[][]{series.asArray()},
                new double[][]{
                        forecast.pointEstimates().asArray(),
                        forecast.lowerPredictionInterval().asArray(),
                        forecast.upperPredictionInterval().asArray()
                }
        );

        JFreeChart chart = ChartFactory.createXYLineChart(
                title, // Chart title
                "Key", // X-Axis Label
                "Value", // Y-Axis Label
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
            signafloARIMASeasonalExample example = new signafloARIMASeasonalExample("ARIMA Seasonal");
            example.setAlwaysOnTop(true);
            example.pack();
            example.setSize(1200, 800);
            example.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
            example.setVisible(true);
        });
    }
}
