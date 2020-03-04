package pro.skymind.ts.decomposition.moving_median;

import com.github.manliogit.timeserie.smooth.MovingAverage;
import com.github.manliogit.timeserie.smooth.MovingMedian;
import com.github.signaflo.timeseries.TestData;
import com.github.signaflo.timeseries.TimeSeries;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.AxisLocation;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.NumberTickUnit;
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

/**
 * reference: https://github.com/manlioGit/time-serie/blob/master/src/test/java/com/github/manliogit/timeserie/SerieTest.java#L185
 */
public class timeserieMovingMedianSubPlot extends ApplicationFrame {
    /**
     * Constructs a new demonstration application.
     *
     * @param title the frame title.
     */
    public timeserieMovingMedianSubPlot(final String title) {

        super(title);

        TimeSeries data = TestData.elecSales;
        MovingMedian MA3 = new MovingMedian(data.asList(), 3);
        MovingMedian MA5 = new MovingMedian(data.asList(), 5);
        MovingMedian MA7 = new MovingMedian(data.asList(), 7);
        MovingMedian MA9 = new MovingMedian(data.asList(), 9);

        final JFreeChart chart = createCombinedChart(
                new String[]{"elecSales", "MMed3", "MMed5", "MMed7", "MMed9"},
                new double[][]{
                        data.asList().stream().mapToDouble(Double::doubleValue).toArray(),
                        MA3.trend().stream().mapToDouble(Double::doubleValue).toArray(),
                        MA5.trend().stream().mapToDouble(Double::doubleValue).toArray(),
                        MA7.trend().stream().mapToDouble(Double::doubleValue).toArray(),
                        MA9.trend().stream().mapToDouble(Double::doubleValue).toArray()
                },
                new int[]{0, 3/2, 5/2, 7/2, 9/2}
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
            rangeAxis.setRange(2000, 4000);
            rangeAxis.setTickUnit(new NumberTickUnit(500));
            final XYPlot subplot = new XYPlot(data, null, rangeAxis, renderer);
            subplot.setRangeAxisLocation(AxisLocation.BOTTOM_OR_LEFT);
            plot.add(subplot, 1);
        }

        return new JFreeChart(timeserieMovingMedianSubPlot.class.getCanonicalName(), JFreeChart.DEFAULT_TITLE_FONT, plot, true);
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
        final timeserieMovingMedianSubPlot demo = new timeserieMovingMedianSubPlot("Moving Median");
        demo.pack();
        RefineryUtilities.centerFrameOnScreen(demo);
        demo.setVisible(true);
    }
}
