/*
 *
 *  * ******************************************************************************
 *  *  * Copyright (c) 2020 CertifAI Sdn. Bhd.
 *  *  *
 *  *  * This program and the accompanying materials are made available under the
 *  *  * terms of the Apache License, Version 2.0 which is available at
 *  *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *  *
 *  *  * Unless required by applicable law or agreed to in writing, software
 *  *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  *  * License for the specific language governing permissions and limitations
 *  *  * under the License.
 *  *  *
 *  *  * SPDX-License-Identifier: Apache-2.0
 *  *  *****************************************************************************
 *
 *
 */

package ai.certifai.training.dl2_regression;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.DatasetRenderingOrder;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.GrayPaintScale;
import org.jfree.chart.renderer.PaintScale;
import org.jfree.chart.renderer.xy.XYBlockRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.DefaultXYDataset;
import org.jfree.data.xy.XYDataset;
import org.jfree.ui.RectangleInsets;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.swing.*;
import java.awt.*;

/**
 * Regression scatterplot (actual vs fitted)
 *
 */
public class PlotUtil {


    /**
     * Create data for the background data set
     */
    private static XYDataset createBackgroundData(INDArray backgroundIn, INDArray backgroundOut) {
        int nRows = backgroundIn.rows();
        double[] xValues = new double[nRows];
        double[] yValues = new double[nRows];

        for (int i = 0; i < nRows; i++) {
            xValues[i] = backgroundIn.getDouble(i, 0);
            yValues[i] = backgroundOut.getDouble(i, 0);

        }

        DefaultXYDataset dataset = new DefaultXYDataset();
        dataset.addSeries("Series 1",
                new double[][]{xValues, yValues});

        return dataset;
    }

    private static JFreeChart createChart(XYDataset dataset, double[] mins, double[] maxs, int nPoints, XYDataset xyData) {
        NumberAxis xAxis = new NumberAxis("labels");
        xAxis.setRange(mins[0], maxs[0]);

        NumberAxis yAxis = new NumberAxis("pred");
        yAxis.setRange(mins[1], maxs[1]);

        XYBlockRenderer renderer = new XYBlockRenderer();
        renderer.setBlockWidth((maxs[0] - mins[0]) / (nPoints - 1));
        renderer.setBlockHeight((maxs[1] - mins[1]) / (nPoints - 1));
        PaintScale scale = new GrayPaintScale(0, 1.0);
        renderer.setPaintScale(scale);
        XYPlot plot = new XYPlot(dataset, xAxis, yAxis, renderer);
        plot.setBackgroundPaint(Color.lightGray);
        plot.setDomainGridlinesVisible(false);
        plot.setRangeGridlinesVisible(false);
        plot.setAxisOffset(new RectangleInsets(5, 5, 5, 5));
        JFreeChart chart = new JFreeChart("", plot);
        chart.getXYPlot().getRenderer().setSeriesVisibleInLegend(0, false);

        ChartUtilities.applyCurrentTheme(chart);

        plot.setDataset(1, xyData);
        XYLineAndShapeRenderer renderer2 = new XYLineAndShapeRenderer();
        renderer2.setBaseLinesVisible(false);
        plot.setRenderer(1, renderer2);

        plot.setDatasetRenderingOrder(DatasetRenderingOrder.FORWARD);

        return chart;
    }

    private static void visualizeRegression (INDArray labels, INDArray pred ) {

        double pred_min = pred.min(0).data().asDouble()[0];
        double label_min = labels.min(0).data().asDouble()[0];
        double pred_max = pred.max(0).data().asDouble()[0];
        double label_max = labels.max(0).data().asDouble()[0];

        // Make sure both axes have same min values
        double[] mins;
        if (pred_min > label_min) {
            mins = new double[]{pred_min, pred_min};
        } else {
            mins = new double[]{label_min, label_min};
        }


        // Make sure both axes have same max values
        double[] maxs;
        if (pred_max > label_max) {
            maxs = new double[]{pred_max, pred_max};
        } else {
            maxs = new double[]{label_max, label_max};
        }

        XYDataset backgroundData = createBackgroundData(labels, pred);
        JPanel panel = new ChartPanel(createChart(backgroundData, mins, maxs, 400, backgroundData));

        JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();
        f.setTitle("Prediction vs Labels");

        f.setVisible(true);
        f.setLocationRelativeTo(null);
    }

}
