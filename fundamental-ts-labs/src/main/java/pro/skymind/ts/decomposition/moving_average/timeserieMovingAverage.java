package pro.skymind.ts.decomposition.moving_average;

import com.github.manliogit.timeserie.smooth.MovingAverage;
import com.github.signaflo.timeseries.TestData;
import com.github.signaflo.timeseries.TimeSeries;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.NumberTickUnit;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;

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

        TimeSeries data = TestData.elecSales;
        MovingAverage MA3 = new MovingAverage(data.asList(), 3);
        MovingAverage MA5 = new MovingAverage(data.asList(), 5);
        MovingAverage MA7 = new MovingAverage(data.asList(), 7);
        MovingAverage MA9 = new MovingAverage(data.asList(), 9);

        final JFreeChart chart = createCombinedChart(
                new String[]{"elecSales", "MA3", "MA5", "MA7", "MA9"},
                new double[][]{
                        data.asList().stream().mapToDouble(Double::doubleValue).toArray(),
                        MA3.trend().stream().mapToDouble(Double::doubleValue).toArray(),
                        MA5.trend().stream().mapToDouble(Double::doubleValue).toArray(),
                        MA7.trend().stream().mapToDouble(Double::doubleValue).toArray(),
                        MA9.trend().stream().mapToDouble(Double::doubleValue).toArray()
                },
                new int[]{1/2, 3/2, 5/2, 7/2, 9/2}
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

        XYSeriesCollection dataset = new XYSeriesCollection();
        for (int i = 0; i < seriesNames.length; i++) {
            XYSeries series = new XYSeries(String.valueOf(seriesNames[i]));
            for (int j = 0; j < seriesValues[i].length; j++) {
                series.add(j + startIndex[i], seriesValues[i][j]);
            }
            dataset.addSeries(series);
        }
        JFreeChart chart = ChartFactory.createXYLineChart(
                "Moving Average", // Chart title
                "Month", // X-Axis Label
                "elecSales", // Y-Axis Label
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                true
        );

        NumberAxis range = (NumberAxis) chart.getXYPlot().getRangeAxis();
        range.setRange(2000, 4000);
        range.setTickUnit(new NumberTickUnit(200));
        return chart;
    }

    /**
     * Starting point for the demonstration application.
     *
     * @param args ignored.
     */
    public static void main(final String[] args) {
        final timeserieMovingAverage demo = new timeserieMovingAverage("Moving Average");
        demo.pack();
        RefineryUtilities.centerFrameOnScreen(demo);
        demo.setVisible(true);
    }
}
