package pro.skymind.ts.arima;

import com.github.signaflo.timeseries.TestData;
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

import static com.github.signaflo.data.visualization.Plots.plot;

public class signafloARIMADebitCards extends ApplicationFrame {

    /**
     * Creates a new demo.
     *
     * @param title the frame title.
     */
    public signafloARIMADebitCards(String title) {
        super(title);

        TimeSeries debitcards = TestData.debitcards;

        ArimaOrder modelOrder = ArimaOrder.order(0, 1, 1, 0, 1, 1); // Note that intercept fitting will automatically be turned off

        com.github.signaflo.timeseries.model.arima.Arima model = Arima.model(debitcards, modelOrder);

        System.out.println(model.aic()); // Get and display the model AIC
        System.out.println(model.coefficients()); // Get and display the estimated coefficients
        System.out.println(java.util.Arrays.toString(model.stdErrors()));

//        plot(model.predictionErrors());

        int forecast_steps = 12;
        Forecast forecast = model.forecast(forecast_steps); // To specify the alpha significance level, add it as a second argument.

        System.out.println(forecast);


        YIntervalSeriesCollection collection = new YIntervalSeriesCollection();
        XYDataset dataset = createYIntervalSeries(
                collection,
                "Data",
                debitcards.asArray(),
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
                signafloARIMADebitCards.class.getCanonicalName(),      // chart title
                "Month",                      // x axis label
                "Amount",                      // y axis label
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
        renderer.setSeriesStroke(0, new BasicStroke(1.0f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
        renderer.setSeriesStroke(0, new BasicStroke(1.0f));
        renderer.setSeriesFillPaint(0, Color.WHITE);
        renderer.setSeriesStroke(1, new BasicStroke(1.0f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
        renderer.setSeriesStroke(1, new BasicStroke(1.0f));
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

        signafloARIMADebitCards demo = new signafloARIMADebitCards("ARIMA Debit Cards");
        demo.pack();
        RefineryUtilities.centerFrameOnScreen(demo);
        demo.setVisible(true);

    }

}