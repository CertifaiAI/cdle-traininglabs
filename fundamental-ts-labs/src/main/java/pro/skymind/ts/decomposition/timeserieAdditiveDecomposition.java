package pro.skymind.ts.decomposition;

import com.github.manliogit.timeserie.Serie;
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

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.List;

public class timeserieAdditiveDecomposition extends ApplicationFrame {
    /**
     * Constructs a new demonstration application.
     *
     * @param title the frame title.
     */
    public timeserieAdditiveDecomposition(final String title) {

        super(title);

        int movingAverageWindow = 4;
        Serie serie = new Serie(ausBeer(), movingAverageWindow);

        final JFreeChart chart = createCombinedChart(
                new String[]{"airPassengers", "MA / Trend", "Season", "Residual"},
                new double[][]{
                        ausBeer().stream().mapToDouble(Double::doubleValue).toArray(),
                        serie.trend().stream().mapToDouble(Double::doubleValue).toArray(),
                        VisualizerHelper.repeat(serie.season().stream().mapToDouble(Double::doubleValue).toArray(), ausBeer().size()),
                        serie.residual().stream().mapToDouble(Double::doubleValue).toArray()
                },
                new int[]{0, movingAverageWindow, 0, movingAverageWindow}
        );

        final ChartPanel panel = new ChartPanel(chart, true, true, true, false, true);
        panel.setPreferredSize(new java.awt.Dimension(1000, 600));
        setContentPane(panel);
    }

    private List<Double> ausBeer() {
        return Arrays.asList(236.,320.,272.,233.,237.,313.,261.,227.,250.,314.,286.,227.,260.,311.,295.,233.,257.,339.,279.,250.,270.,346.,294.,255.,278.,363.,313.,273.,300.,370.,331.,288.,306.,386.,335.,288.,308.,402.,353.,316.,325.,405.,393.,319.,327.,442.,383.,332.,361.,446.,387.,357.,374.,466.,410.,370.,379.,487.,419.,378.,393.,506.,458.,387.);
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

        return new JFreeChart(timeserieAdditiveDecomposition.class.getCanonicalName(), JFreeChart.DEFAULT_TITLE_FONT, plot, true);
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

        final timeserieAdditiveDecomposition demo = new timeserieAdditiveDecomposition("Multiplicative Decomposition");
        demo.pack();
        RefineryUtilities.centerFrameOnScreen(demo);
        demo.setVisible(true);

    }
}
