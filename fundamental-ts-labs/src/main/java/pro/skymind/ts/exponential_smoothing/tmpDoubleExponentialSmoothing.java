package pro.skymind.ts.exponential_smoothing;

import java.util.Arrays;

public class tmpDoubleExponentialSmoothing {

    /**
     * Performs double exponential smoothing for given time series.
     * <p/>
     * This method is suitable for fitting series with linear trend.
     *
     * @param data  An array containing the recorded data of the time series
     * @param alpha Smoothing factor for data (0 < alpha < 1)
     * @param beta  Smoothing factor for trend (0 < beta < 1)
     * @return Instance of model that can be used to forecast future values
     */
    public static Model fit(double[] data, double alpha, double beta) {
        validateParams(alpha, beta);                        //validating values of alpha and beta

        double[] smoothedData = new double[data.length];    //array to store smoothed values

        double[] trends = new double[data.length + 1];
        double[] levels = new double[data.length + 1];

        //initializing values of parameters
        smoothedData[0] = data[0];
        trends[0] = data[1] - data[0];
        levels[0] = data[0];

        for (int t = 0; t < data.length; t++) {
            smoothedData[t] = trends[t] + levels[t];
            levels[t + 1] = alpha * data[t] + (1 - alpha) * (levels[t] + trends[t]);
            trends[t + 1] = beta * (levels[t + 1] - levels[t]) + (1 - beta) * trends[t];
        }
        return new Model(smoothedData, trends, levels, calculateSSE(data, smoothedData));
    }

    private static double calculateSSE(double[] data, double[] smoothedData) {
        double sse = 0;
        for (int i = 0; i < data.length; i++) {
            sse += Math.pow(smoothedData[i] - data[i], 2);
        }
        return sse;
    }

    private static void validateParams(double alpha, double beta) {
        if (alpha < 0 || alpha > 1) {
            throw new RuntimeException("The value of alpha must be between 0 and 1");
        }

        if (beta < 0 || beta > 1) {
            throw new RuntimeException("The value of beta must be between 0 and 1");
        }
    }

    static class Model {
        private final double[] smoothedData;
        private final double[] trends;
        private final double[] levels;
        private final double sse;

        public Model(double[] smoothedData, double[] trends, double[] levels, double sse) {
            this.smoothedData = smoothedData;
            this.trends = trends;
            this.levels = levels;
            this.sse = sse;
        }

        /**
         * Forecasts future values.
         *
         * @param size no of future values that you need to forecast
         * @return forecast data
         */
        double[] forecast(int size) {
            double[] forecastData = new double[size];
            for (int i = 0; i < size; i++) {
                forecastData[i] = levels[levels.length - 1] + (i + 1) * trends[trends.length - 1];
            }
            return forecastData;
        }

        public double[] getSmoothedData() {
            return smoothedData;
        }

        public double[] getTrend() {
            return trends;
        }

        public double[] getLevel() {
            return levels;
        }

        public double getSSE() {
            return sse;
        }
    }
}