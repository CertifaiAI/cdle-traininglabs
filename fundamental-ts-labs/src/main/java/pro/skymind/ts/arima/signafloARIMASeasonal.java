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
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.DeviationRenderer;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.YIntervalSeries;
import org.jfree.data.xy.YIntervalSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RectangleInsets;
import org.jfree.ui.RefineryUtilities;

import javax.swing.*;
import java.awt.*;

public class signafloARIMASeasonal extends ApplicationFrame {

    /**
     * Creates a new demo.
     *
     * @param title the frame title.
     */
    public signafloARIMASeasonal(String title) {
        super(title);

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

        com.github.signaflo.timeseries.model.arima.Arima model = Arima.model(series, order, week);

        // Finally, generate a forecast for next week using the model
        Forecast forecast = model.forecast(7);
        System.out.println(forecast);

        YIntervalSeriesCollection collection = new YIntervalSeriesCollection();
        XYDataset dataset = createYIntervalSeries(
                collection,
                "Data",
                series.asArray(),
                "Forecast",
                forecast.pointEstimates().asArray(),
                forecast.lowerPredictionInterval().asArray(),
                forecast.upperPredictionInterval().asArray()
        );
        JFreeChart chart = createChart(dataset);
        JPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new Dimension(1000, 600));
        setContentPane(chartPanel);
    }

    /**
     * Creates a sample dataset.
     *
     * @param collection
     * @param dataSeriesName
     * @param data
     * @param forecastSeriesName
     * @param forecast
     * @param lower
     * @param upper
     * @return a sample dataset.
     */
    private static XYDataset createYIntervalSeries(YIntervalSeriesCollection collection, String dataSeriesName, double[] data, String forecastSeriesName, double[] forecast, double[] lower, double[] upper) {
        YIntervalSeries dseries = new YIntervalSeries(dataSeriesName);
        for (int i = 0; i < data.length; i++) {
            dseries.add(i, data[i], data[i], data[i]);
        }
        collection.addSeries(dseries);

        YIntervalSeries fseries = new YIntervalSeries(forecastSeriesName);
        for (int i = 0; i < forecast.length; i++) {
            // additional plot for better visualization
            if (i == 0)
                fseries.add(data.length - 1, data[data.length - 1], data[data.length - 1], data[data.length - 1]);

            fseries.add(i + data.length, forecast[i], lower[i], upper[i]);
        }
        collection.addSeries(fseries);
        return collection;
    }

    /**
     * Creates a chart.
     *
     * @param dataset the data for the chart.
     * @return a chart.
     */
    private static JFreeChart createChart(XYDataset dataset) {

        // create the chart...
        JFreeChart chart = ChartFactory.createXYLineChart(
                signafloARIMASeasonal.class.getCanonicalName(),      // chart title
                "X",                      // x axis label
                "Y",                      // y axis label
                dataset,                  // data
                PlotOrientation.VERTICAL,
                true,                     // include legend
                true,                     // tooltips
                false                     // urls
        );

        chart.setBackgroundPaint(Color.white);

        // get a reference to the plot for further customisation...
        XYPlot plot = (XYPlot) chart.getPlot();
        plot.setBackgroundPaint(Color.lightGray);
        plot.setAxisOffset(new RectangleInsets(5.0, 5.0, 5.0, 5.0));
        plot.setDomainGridlinePaint(Color.white);
        plot.setRangeGridlinePaint(Color.white);

        DeviationRenderer renderer = new DeviationRenderer(true, false);
        renderer.setSeriesStroke(0, new BasicStroke(3.0f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
        renderer.setSeriesStroke(0, new BasicStroke(3.0f));
        renderer.setSeriesFillPaint(0, Color.WHITE);
        renderer.setSeriesStroke(1, new BasicStroke(3.0f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
        renderer.setSeriesStroke(1, new BasicStroke(3.0f));
        renderer.setSeriesFillPaint(1, Color.WHITE);
        plot.setRenderer(renderer);

        // change the auto tick unit selection to integer units only...
        NumberAxis yAxis = (NumberAxis) plot.getRangeAxis();
        yAxis.setAutoRangeIncludesZero(false);
        yAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
        return chart;
    }

    /**
     * Starting point for the demonstration application.
     *
     * @param args ignored.
     */
    public static void main(String[] args) {

        signafloARIMASeasonal demo = new signafloARIMASeasonal("ARIMA Seasonal (Syntactic Data)");
        demo.pack();
        RefineryUtilities.centerFrameOnScreen(demo);
        demo.setVisible(true);

    }

}