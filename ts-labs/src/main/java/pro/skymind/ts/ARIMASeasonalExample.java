package pro.skymind.ts;

import com.github.signaflo.math.stats.distributions.Normal;
import com.github.signaflo.timeseries.TimePeriod;
import com.github.signaflo.timeseries.TimeSeries;
import com.github.signaflo.timeseries.forecast.Forecast;
import com.github.signaflo.timeseries.model.arima.Arima;
import com.github.signaflo.timeseries.model.arima.ArimaOrder;

public class ARIMASeasonalExample {
    public static void main(String[] args) {
        // First, we'll fill in 15 weeks worth of daily data with an extremely simple
        // simulated data generating process.
        Normal normal = new Normal(); // Create a normal distribution with mean 0 an sd of 1.

        double[] values = new double[105];
        for (int i = 0; i < values.length; i++) {
            values[i] = normal.rand();
        }

        // Assumes Monday corresponds to 0.
        for (int fri = 4; fri < values.length; fri += 7) {
            values[fri] += 1.0;
            values[fri + 1] += 2.0;
            values[fri + 2] -= 1.0;
        }

        // Second, we'll create a daily time series from those values.
        TimePeriod day = TimePeriod.oneDay();

        TimeSeries series = TimeSeries.from(day, values);

        // Third, we'll create an ArimaOrder with a seasonal component.
        ArimaOrder order = ArimaOrder.order(0, 0, 0, 1, 1, 1);

        // Fourth, we create an ARIMA model with the series, the order,
        // and the weekly seasonality.
        TimePeriod week = TimePeriod.oneWeek();

        Arima model = Arima.model(series, order, week);

        // Finally, generate a forecast for next week using the model
        Forecast forecast = model.forecast(7);
        System.out.println(forecast);
    }
}
