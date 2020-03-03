package pro.skymind.ts.decomposition;

import com.github.manliogit.timeserie.Serie;
import org.apache.commons.lang3.ArrayUtils;
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
 * reference: https://github.com/manlioGit/time-serie/blob/master/src/test/java/com/github/manliogit/timeserie/SerieTest.java#L160
 */
public class timeserieMultiplicativeDecomposition extends ApplicationFrame {
    /**
     * Constructs a new demonstration application.
     *
     * @param title the frame title.
     */
    public timeserieMultiplicativeDecomposition(final String title) {

        super(title);


        List<Double> airPassengers = Arrays.asList(112., 118., 132., 129., 121., 135., 148., 148., 136., 119., 104., 118., 115., 126., 141., 135., 125., 149., 170., 170., 158., 133., 114., 140., 145., 150., 178., 163., 172., 178., 199., 199., 184., 162., 146., 166., 171., 180., 193., 181., 183., 218., 230., 242., 209., 191., 172., 194., 196., 196., 236., 235., 229., 243., 264., 272., 237., 211., 180., 201., 204., 188., 235., 227., 234., 264., 302., 293., 259., 229., 203., 229., 242., 233., 267., 269., 270., 315., 364., 347., 312., 274., 237., 278., 284., 277., 317., 313., 318., 374., 413., 405., 355., 306., 271., 306., 315., 301., 356., 348., 355., 422., 465., 467., 404., 347., 305., 336., 340., 318., 362., 348., 363., 435., 491., 505., 404., 359., 310., 337., 360., 342., 406., 396., 420., 472., 548., 559., 463., 407., 362., 405., 417., 391., 419., 461., 472., 535., 622., 606., 508., 461., 390., 432.);

        int movingAverageWindow = 12;
        Serie serie = new Serie(airPassengers, movingAverageWindow).multiplicative();

        final JFreeChart chart = createCombinedChart(
                new String[]{"airPassengers", "MA / Trend", "Season", "Residual"},
                new double[][]{
                        airPassengers.stream().mapToDouble(Double::doubleValue).toArray(),
                        serie.trend().stream().mapToDouble(Double::doubleValue).toArray(),
                        VisualizerHelper.repeat(serie.season().stream().mapToDouble(Double::doubleValue).toArray(), airPassengers.size()),
                        serie.residual().stream().mapToDouble(Double::doubleValue).toArray()
                },
                new int[]{0, movingAverageWindow, 0, movingAverageWindow}
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

        return new JFreeChart(timeserieMultiplicativeDecomposition.class.getCanonicalName(), JFreeChart.DEFAULT_TITLE_FONT, plot, true);
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

        final timeserieMultiplicativeDecomposition demo = new timeserieMultiplicativeDecomposition("Multiplicative Decomposition");
        demo.pack();
        RefineryUtilities.centerFrameOnScreen(demo);
        demo.setVisible(true);

    }
}
