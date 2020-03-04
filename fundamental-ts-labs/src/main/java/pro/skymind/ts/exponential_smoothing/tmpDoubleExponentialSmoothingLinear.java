package pro.skymind.ts.exponential_smoothing;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.AxisLocation;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.CombinedDomainXYPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.StandardXYItemRenderer;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;

import java.util.Arrays;

/**
 * reference: https://github.com/manlioGit/time-serie/blob/master/src/test/java/com/github/manliogit/timeserie/SerieTest.java#L185
 */
public class tmpDoubleExponentialSmoothingLinear extends ApplicationFrame {
    /**
     * Constructs a new demonstration application.
     *
     * @param title the frame title.
     */
    public tmpDoubleExponentialSmoothingLinear(final String title) {

        super(title);

        double[] testData = {17.55, 21.86, 23.89, 26.93, 26.89, 28.83, 30.08, 30.95, 30.19, 31.58, 32.58, 33.48, 39.02, 41.39, 41.60};

        tmpDoubleExponentialSmoothing.Model model = tmpDoubleExponentialSmoothing.fit(testData, 0.8, 0.2);

        System.out.println("Input values: " + Arrays.toString(testData));
        System.out.println("Smoothed values: " + Arrays.toString(model.getSmoothedData()));
        System.out.println("Trend: " + Arrays.toString(model.getTrend()));
        System.out.println("Level: " + Arrays.toString(model.getLevel()));
        System.out.println("Forecast: " + Arrays.toString(model.forecast(5)));
        System.out.println("Sum of squared error: " + model.getSSE());


        final JFreeChart chart = createCombinedChart(
                new String[]{"Data", "SmoothedData", "Trend", "Level", "Forecast"},
                new double[][]{
                        testData,
                        model.getSmoothedData(),
                        model.getTrend(),
                        model.getLevel(),
                        model.forecast(5)
                },
                new int[]{0, 0, 0, 0, 0}
        );

        final ChartPanel panel = new ChartPanel(chart, true, true, true, false, true);
        panel.setPreferredSize(new java.awt.Dimension(1000, 600));
        setContentPane(panel);
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

        // parent plot...
        final CombinedDomainXYPlot plot = new CombinedDomainXYPlot(new NumberAxis("Time"));
        plot.setGap(10.0);
        plot.setOrientation(PlotOrientation.VERTICAL);

        for (int i = 0; i < seriesNames.length; i++) {
            // create subplot 1...
            final XYDataset data = createDataset(seriesNames[i], seriesValues[i], startIndex[i]);
            final XYItemRenderer renderer = new StandardXYItemRenderer();
            final NumberAxis rangeAxis = new NumberAxis(seriesNames[i]);
//            rangeAxis.setRange(2000, 4000);
//            rangeAxis.setTickUnit(new NumberTickUnit(500));
            final XYPlot subplot = new XYPlot(data, null, rangeAxis, renderer);
            subplot.setRangeAxisLocation(AxisLocation.BOTTOM_OR_LEFT);
            plot.add(subplot, 1);
        }

        return new JFreeChart(tmpDoubleExponentialSmoothingLinear.class.getCanonicalName(), JFreeChart.DEFAULT_TITLE_FONT, plot, true);
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
        final tmpDoubleExponentialSmoothingLinear demo = new tmpDoubleExponentialSmoothingLinear("Moving Average");
        demo.pack();
        RefineryUtilities.centerFrameOnScreen(demo);
        demo.setVisible(true);
    }
}
