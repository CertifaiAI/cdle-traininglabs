package pro.skymind.ts.arima;

import com.github.signaflo.timeseries.TestData;
import com.github.signaflo.timeseries.TimeSeries;
import com.github.signaflo.timeseries.forecast.Forecast;
import com.github.signaflo.timeseries.model.arima.Arima;
import com.github.signaflo.timeseries.model.arima.ArimaOrder;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYDataset;
import pro.skymind.ts.decomposition.VisualizerHelper;

import javax.swing.*;

import static com.github.signaflo.data.visualization.Plots.plot;

/**
 * reference: https://github.com/signaflo/java-timeseries/wiki/ARIMA%20models
 */
public class signafloARIMADebitCards extends JFrame {

    public signafloARIMADebitCards(String title) {
        TimeSeries debitcards = TestData.debitcards;

        ArimaOrder modelOrder = ArimaOrder.order(0, 1, 1, 0, 1, 1); // Note that intercept fitting will automatically be turned off

        Arima model = Arima.model(debitcards, modelOrder);

        System.out.println(model.aic()); // Get and display the model AIC
        System.out.println(model.coefficients()); // Get and display the estimated coefficients
        System.out.println(java.util.Arrays.toString(model.stdErrors()));

//        plot(model.predictionErrors());

        int forecast_steps = 12;
        Forecast forecast = model.forecast(forecast_steps); // To specify the alpha significance level, add it as a second argument.

        System.out.println(forecast);

        XYDataset dataset = VisualizerHelper.createXYDatasetJoint(
                new String[]{"debitcards"},
                new String[]{"forecast", "lower", "upper"},
                new double[][]{debitcards.asArray()},
                new double[][]{
                        forecast.pointEstimates().asArray(),
                        forecast.lowerPredictionInterval().asArray(),
                        forecast.upperPredictionInterval().asArray()
                }
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
            signafloARIMADebitCards example = new signafloARIMADebitCards("ARIMA");
            example.setAlwaysOnTop(true);
            example.pack();
            example.setSize(1200, 800);
            example.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
            example.setVisible(true);
        });
    }
}
