package ai.certifai.training.dl3_tuning;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;

import javax.swing.*;
import java.util.ArrayList;

public class PlotUtil {

    // function to create the dataset
    private static DefaultCategoryDataset createDataset(ArrayList<Double> trainingScore, ArrayList<Double> validationScore, int numOfEpochs) {

        // create series of dataset
        DefaultCategoryDataset dataset = new DefaultCategoryDataset ();
        String s1 = "Training";
        String s2 = "Validation";

        // add values to the series
        for(int i=0; i<numOfEpochs; i++){
            dataset.addValue(trainingScore.get(i), s1, String.valueOf(i));
            dataset.addValue(validationScore.get(i), s2, String.valueOf(i));
        }

        return dataset;
    }

    // function to create line graph
    private static JFreeChart createGraph(String chartTitle, String XLabel, String YLabel, ArrayList<Double> xValuesList1, ArrayList<Double> xValuesList2, int numOfEpochs){

        // create line chart
        JFreeChart lineChart = ChartFactory.createLineChart(
                chartTitle,
                XLabel,YLabel,
                createDataset(xValuesList1, xValuesList2, numOfEpochs),
                PlotOrientation.VERTICAL,
                true,true,false);

        return lineChart;
    }

    // public function to plot the loss graph
    public static void plotLossGraph(String XLabel, String YLabel, ArrayList<Double> xValuesList1, ArrayList<Double> xValuesList2, int numOfEpochs){

        // call createGraph function to draw the graph
        JFreeChart lineGraph = createGraph("Training/Validation Loss", XLabel, YLabel, xValuesList1, xValuesList2, numOfEpochs);

        // initialise the UI
        ChartPanel chartPanel = new ChartPanel( lineGraph );
        chartPanel.setPreferredSize( new java.awt.Dimension( 560 , 367 ) );
        JFrame f = new JFrame();
        f.add(chartPanel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();
        f.setTitle("Loss Graph");
        f.setVisible(true);
        f.setLocationRelativeTo(null);
    }

}
