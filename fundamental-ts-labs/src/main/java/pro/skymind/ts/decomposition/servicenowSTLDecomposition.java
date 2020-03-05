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
 * Reference: http://netlib.org/a/stl
 * fPeriodLength:
 *          input, the period of the seasonal component. For example,
 *          if the time series is monthly with a yearly cycle, then
 *          np=12.
 * fSeasonalWidth:
 *          input, length of the seasonal smoother. The value of ns
 *          should be an odd integer greater than or equal to 3; ns>6
 *          is recommended. As ns increases the values of the
 *          seasonal component at a given point in the seasonal cycle
 *          (e.g., January values of a monthly series with a yearly
 *          cycle) become smoother.
 * fSeasonalJump:
 *          input, skipping value for seasonal smoothing. The
 *          seasonal smoother skips ahead nsjump points and then
 *          linearly interpolates in between. The value of nsjump
 *          should be a positive integer; if nsjump=1, a seasonal
 *          smooth is calculated at all n points. To make the
 *          procedure run faster, a reasonable choice for nsjump is
 *          10%-20% of ns.
 * fSeasonalDegree:
 *          input, degree of locally-fitted polynomial in seasonal
 *          smoothing. The value is 0 or 1.
 * fTrendWidth:
 *          input, length of the trend smoother. The value of nt
 *          should be an odd integer greater than or equal to 3; a
 *          value of nt between 1.5*np and 2*np is recommended. As
 *          nt increases the values of the trend component become
 *          smoother.
 * fTrendJump:
 *          input, skipping value for trend smoothing.
 * fTrendDegree:
 *          input, degree of locally-fitted polynomial in trend
 *          smoothing. The value is 0 or 1.
 * fLowpassWidth:
 *          input, length of the low-pass filter. The value of nl
 *          should be an odd integer greater than or equal to 3; the
 *          smallest odd integer greater than or equal to np is
 *          recommended.
 * fLowpassJump:
 *          input, skipping value for the low-pass filter.
 * fLowpassDegree (1):
 *          input, degree of locally-fitted polynomial in low-pass
 *          smoothing. The value is 0 or 1.
 * fInnerIterations (2):
 *          input, number of loops for updating the seasonal and
 *          trend components. The value of ni should be a positive
 *          integer. See the next argument for advice on the choice
 *          of ni.
 * fRobustIterations (0):
 *          input, number of iterations of robust fitting. The value
 *          of no should be a nonnegative integer. If the data are
 *          well behaved without outliers, then robustness iterations
 *          are not needed. In this case set no=0, and set ni=2 to 5
 *          depending on how much security you want that the
 *          seasonal-trend looping converges. If outliers are
 *          present then no=3 is a very secure value unless the
 *          outliers are radical, in which case no=5 or even 10 might
 *          be better. If no>0 then set ni to 1 or 2.
 * fPeriodic (false):
 * fFlatTrend (false):
 * fLinearTrend (false):
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


        SeasonalTrendLoess smoother = new SeasonalTrendLoess.Builder()
                .setPeriodLength(12)    // Data has a period of 12
                .setSeasonalWidth(156)   // Monthly data smoothed over 35 years
                .setNonRobust()         // Not expecting outliers, so no robustness iterations
                .buildSmoother(values);

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
     * @param seriesName
     * @param seriesValue
     * @return Series.
     */
    private XYDataset createDataset(String seriesName, double[] seriesValue) {

        // create dataset 1...
        final XYSeries series1 = new XYSeries(seriesName);

        for (int i = 0; i < seriesValue.length; i++) {
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
