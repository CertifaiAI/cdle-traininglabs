package pro.skymind.ts.decomposition;

import com.github.servicenow.ds.stats.stl.SeasonalTrendLoess;
import com.github.signaflo.timeseries.TestData;
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

/**
 * A demonstration application showing how to create a vertical combined chart.
 */
public class servicenowSTLDecomposition extends ApplicationFrame {

    /**
     * Constructs a new demonstration application.
     *
     * @param title the frame title.
     */
    public servicenowSTLDecomposition(final String title) {

        super(title);
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

        final JFreeChart chart = createCombinedChart(
                new String[]{"values", "seasonal", "trend", "residual"},
                new double[][]{values, seasonal, trend, residual}
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
     * @return the combined chart.
     */
    private JFreeChart createCombinedChart(String[] seriesNames, double[][] seriesValues) {

        // parent plot...
        final CombinedDomainXYPlot plot = new CombinedDomainXYPlot(new NumberAxis("Time"));
        plot.setGap(10.0);
        plot.setOrientation(PlotOrientation.VERTICAL);

        for (int i = 0; i < seriesNames.length; i++) {
            // create subplot 1...
            final XYDataset data = createDataset(seriesNames[i], seriesValues[i]);
            final XYItemRenderer renderer = new StandardXYItemRenderer();
            final NumberAxis rangeAxis = new NumberAxis(seriesNames[i]);
            final XYPlot subplot = new XYPlot(data, null, rangeAxis, renderer);
            subplot.setRangeAxisLocation(AxisLocation.BOTTOM_OR_LEFT);
            plot.add(subplot, 1);
        }

        return new JFreeChart(servicenowSTLDecomposition.class.getCanonicalName(), JFreeChart.DEFAULT_TITLE_FONT, plot, true);
    }

    /**
     * Creates a sample dataset.
     *
     * @return Series.
     * @param seriesName
     * @param seriesValue
     */
    private XYDataset createDataset(String seriesName, double[] seriesValue) {

        // create dataset 1...
        final XYSeries series1 = new XYSeries(seriesName);

        for(int i=0;i<seriesValue.length;i++){
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

        final servicenowSTLDecomposition demo = new servicenowSTLDecomposition("STL Decomposition");
        demo.pack();
        RefineryUtilities.centerFrameOnScreen(demo);
        demo.setVisible(true);

    }

}
