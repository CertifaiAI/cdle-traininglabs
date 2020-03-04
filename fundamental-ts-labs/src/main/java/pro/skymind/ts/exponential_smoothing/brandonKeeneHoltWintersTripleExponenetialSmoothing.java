package pro.skymind.ts.exponential_smoothing;

import com.github.signaflo.timeseries.TestData;
import net.sourceforge.openforecast.DataPoint;
import net.sourceforge.openforecast.DataSet;
import net.sourceforge.openforecast.ForecastingModel;
import net.sourceforge.openforecast.Observation;
import net.sourceforge.openforecast.models.SimpleExponentialSmoothingModel;
import org.apache.commons.lang3.ArrayUtils;
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
import org.nd4j.shade.guava.primitives.Doubles;
import org.nd4j.shade.guava.primitives.Ints;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * reference: https://github.com/manlioGit/time-serie/blob/master/src/test/java/com/github/manliogit/timeserie/SerieTest.java#L185
 */
public class brandonKeeneHoltWintersTripleExponenetialSmoothing extends ApplicationFrame {
    /**
     * Constructs a new demonstration application.
     *
     * @param title the frame title.
     */
    public brandonKeeneHoltWintersTripleExponenetialSmoothing(final String title) {

        super(title);

        double[] obs = TestData.debitcards.asArray();

        int[] y = toIntArr(obs);

        double[] beta_0 = HoltWintersTripleExponentialImpl.forecast(y, 0.5, 0, 0, 12, 2, true);
        double[] beta_025 = HoltWintersTripleExponentialImpl.forecast(y, 0.5, 0.25, 0, 12, 2, true);
        double[] beta_05 = HoltWintersTripleExponentialImpl.forecast(y, 0.5, 0.5, 0, 12, 2, true);

        final JFreeChart chart = createCombinedChart(
                new String[]{
                        "observations",
                        "beta:0.0",
                        "beta:0.25",
                        "beta:0.5"
                },
                new double[][]{
                        obs,
                        beta_0,
                        beta_025,
                        beta_05
                },
                new int[]{0, 0, 0, 0}
        );

        final ChartPanel panel = new ChartPanel(chart, true, true, true, false, true);
        panel.setPreferredSize(new java.awt.Dimension(800, 600));
        setContentPane(panel);
    }

    private int[] toIntArr(double[] obs) {
        return Arrays.asList(ArrayUtils.toObject(obs)).stream().mapToInt(i -> i.intValue()).toArray();
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
                "debitcards", // Chart title
                "Month", // X-Axis Label
                "Usage", // Y-Axis Label
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                true
        );

//        NumberAxis range = (NumberAxis) chart.getXYPlot().getRangeAxis();
//        range.setRange(400, 550);
//        range.setTickUnit(new NumberTickUnit(20));
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
        final brandonKeeneHoltWintersTripleExponenetialSmoothing demo = new brandonKeeneHoltWintersTripleExponenetialSmoothing("debitcards");
        demo.pack();
        RefineryUtilities.centerFrameOnScreen(demo);
        demo.setVisible(true);
    }
}
