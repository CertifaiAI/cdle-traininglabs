package pro.skymind.ts.exponential_smoothing;

import net.sourceforge.openforecast.DataPoint;
import net.sourceforge.openforecast.DataSet;
import net.sourceforge.openforecast.ForecastingModel;
import net.sourceforge.openforecast.Observation;
import net.sourceforge.openforecast.models.SimpleExponentialSmoothingModel;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.NumberTickUnit;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;

/**
 * reference: https://github.com/manlioGit/time-serie/blob/master/src/test/java/com/github/manliogit/timeserie/SerieTest.java#L185
 */
public class openforecastSimpleExponentialSmoothing extends ApplicationFrame {
    /**
     * Constructs a new demonstration application.
     *
     * @param title the frame title.
     */
    public openforecastSimpleExponentialSmoothing(final String title) {

        super(title);

        double[] observations = {446.6565, 454.4733, 455.663, 423.6322, 456.2713, 440.5881, 425.3325, 485.1494, 506.0482, 526.792, 514.2689, 494.211};
        DataSet observedData = getDataPoints(observations);

        // SES with alpha 0.2
        DataSet fcSeries_02 = getMockDataPoints(observations.length);
        ForecastingModel model_02 = new SimpleExponentialSmoothingModel(0.2, SimpleExponentialSmoothingModel.HUNTER);
        model_02.init(observedData);
        DataSet alpha_02 = model_02.forecast(fcSeries_02);

        // SES with alpha 0.6
        DataSet fcSeries_06 = getMockDataPoints(observations.length);
        ForecastingModel model_06 = new SimpleExponentialSmoothingModel(0.6, SimpleExponentialSmoothingModel.HUNTER);
        model_06.init(observedData);
        DataSet alpha_06 = model_06.forecast(fcSeries_06);

        // SES with alpha 0.8919987280
        DataSet fcSeries_089 = getMockDataPoints(observations.length);
        ForecastingModel model_089 = new SimpleExponentialSmoothingModel(0.8919987280, SimpleExponentialSmoothingModel.HUNTER);
        model_089.init(observedData);
        DataSet alpha_089 = model_089.forecast(fcSeries_089);

        final JFreeChart chart = createCombinedChart(
                new String[]{
                        "Observations",
                        "alpha 0.2",
                        "alpha 0.6",
                        "alpha 0.89"
                },
                new double[][]{
                        observations,
                        getDoubles(alpha_02),
                        getDoubles(alpha_06),
                        getDoubles(alpha_089)
                },
                new int[]{0, 0, 0, 0}
        );

        final ChartPanel panel = new ChartPanel(chart, true, true, true, false, true);
        panel.setPreferredSize(new java.awt.Dimension(800, 600));
        setContentPane(panel);
    }

    private double[] getDoubles(DataSet results) {
        double[] d = new double[results.size()];
        DataPoint dp;
        for (int t = 0; t < results.size(); t++) {
            dp = (Observation) results.toArray()[t];
            d[t] = dp.getDependentValue();
        }
        return d;
    }

    private DataSet getMockDataPoints(int len) {
        DataSet fcValues = new DataSet();
        DataPoint dp;
        for (int t = 0; t < len; t++) {
            dp = new Observation(t);
            dp.setIndependentValue("t", t);
            fcValues.add(dp);
        }
        return fcValues;
    }

    private DataSet getDataPoints(double[] observations) {
        DataSet d = new DataSet();
        DataPoint dp;
        for (int t = 0; t < observations.length; t++) {
            dp = new Observation(observations[t]);
            dp.setIndependentValue("t", t);
            d.add(dp);
        }
        return d;
    }

    /**
     * Creates a combined chart.
     *
     * @param seriesNames
     * @param seriesValues
     * @param startIndex
     * @return the combined chart.
     */
    private JFreeChart createCombinedChart(String[] seriesNames, double[][] seriesValues, int[] startIndex) {

        XYSeriesCollection dataset = new XYSeriesCollection();
        for (int i = 0; i < seriesNames.length; i++) {
            XYSeries series = new XYSeries(String.valueOf(seriesNames[i]));
            for (int j = 0; j < seriesValues[i].length; j++) {
                series.add(j + startIndex[i], seriesValues[i][j]);
            }
            dataset.addSeries(series);
        }
        JFreeChart chart = ChartFactory.createXYLineChart(
                "Oil production in Saudi Arabia from 1996 to 2007", // Chart title
                "Year", // X-Axis Label
                "Oil (millions of tonnes)", // Y-Axis Label
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                true
        );

        NumberAxis range = (NumberAxis) chart.getXYPlot().getRangeAxis();
        range.setRange(400, 550);
        range.setTickUnit(new NumberTickUnit(20));
        return chart;
    }

    /**
     * Creates a sample dataset.
     *
     * @param seriesName
     * @param seriesValue
     * @param startIndex
     * @return Series.
     */
    private XYDataset createDataset(String seriesName, double[] seriesValue, int startIndex) {

        // create dataset 1...
        final XYSeries series1 = new XYSeries(seriesName);

        for (int i = 0; i < seriesValue.length; i++) {
            series1.add(i + startIndex, seriesValue[i]);
        }

        final XYSeriesCollection collection = new XYSeriesCollection();
        collection.addSeries(series1);

        return collection;
    }

    /**
     * Starting point for the demonstration application.
     *
     * @param args ignored.
     */
    public static void main(final String[] args) {
        final openforecastSimpleExponentialSmoothing demo = new openforecastSimpleExponentialSmoothing("Oil production in Saudi Arabia from 1996 to 2007.");
        demo.pack();
        RefineryUtilities.centerFrameOnScreen(demo);
        demo.setVisible(true);
    }
}
