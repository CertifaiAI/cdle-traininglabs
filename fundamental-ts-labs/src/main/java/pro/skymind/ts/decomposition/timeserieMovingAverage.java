package pro.skymind.ts.decomposition;

import com.github.manliogit.timeserie.Serie;
import com.github.manliogit.timeserie.smooth.MovingAverage;
import com.github.signaflo.timeseries.TestData;
import com.github.signaflo.timeseries.TimeSeries;
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
import pro.skymind.ts.utilities.VisualizerHelper;

import java.util.Arrays;
import java.util.List;

/**
 * reference: https://github.com/manlioGit/time-serie/blob/master/src/test/java/com/github/manliogit/timeserie/SerieTest.java#L185
 */
public class timeserieMovingAverage extends ApplicationFrame {
    /**
     * Constructs a new demonstration application.
     *
     * @param title the frame title.
     */
    public timeserieMovingAverage(final String title) {

        super(title);

        TimeSeries data = TestData.sydneyAir;
        MovingAverage MA2 = new MovingAverage(data.asList(), 2);
        MovingAverage MA4 = new MovingAverage(data.asList(), 4);
        MovingAverage MA8 = new MovingAverage(data.asList(), 8);
        MovingAverage MA16 = new MovingAverage(data.asList(), 16);

        final JFreeChart chart = createCombinedChart(
                new String[]{"sydneyAir", "MA2", "MA4", "MA8", "MA16"},
                new double[][]{
                        data.asList().stream().mapToDouble(Double::doubleValue).toArray(),
                        MA2.trend().stream().mapToDouble(Double::doubleValue).toArray(),
                        MA4.trend().stream().mapToDouble(Double::doubleValue).toArray(),
                        MA8.trend().stream().mapToDouble(Double::doubleValue).toArray(),
                        MA16.trend().stream().mapToDouble(Double::doubleValue).toArray()
                },
                new int[]{0, 2, 4, 8, 16}
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
            final XYPlot subplot = new XYPlot(data, null, rangeAxis, renderer);
            subplot.setRangeAxisLocation(AxisLocation.BOTTOM_OR_LEFT);
            plot.add(subplot, 1);
        }

        return new JFreeChart(timeserieMovingAverage.class.getCanonicalName(), JFreeChart.DEFAULT_TITLE_FONT, plot, true);
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

        for (int i = startIndex; i < seriesValue.length; i++) {
            series1.add(i, seriesValue[i]);
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
        final timeserieMovingAverage demo = new timeserieMovingAverage("Multiplicative Decomposition");
        demo.pack();
        RefineryUtilities.centerFrameOnScreen(demo);
        demo.setVisible(true);
    }
}
