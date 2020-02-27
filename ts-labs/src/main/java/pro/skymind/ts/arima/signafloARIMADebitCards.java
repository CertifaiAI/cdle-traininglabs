package pro.skymind.ts.arima;

import com.github.signaflo.timeseries.TestData;
import com.github.signaflo.timeseries.TimeSeries;
import com.github.signaflo.timeseries.forecast.Forecast;
import com.github.signaflo.timeseries.model.arima.Arima;
import com.github.signaflo.timeseries.model.arima.ArimaOrder;

import static com.github.signaflo.data.visualization.Plots.plot;

/**
 * reference: https://github.com/signaflo/java-timeseries/wiki/ARIMA%20models
 */
public class signafloARIMADebitCards {

    public static void main(String[] args){

        TimeSeries timeSeries = TestData.debitcards;

        ArimaOrder modelOrder = ArimaOrder.order(0, 1, 1, 0, 1, 1); // Note that intercept fitting will automatically be turned off

        Arima model = Arima.model(timeSeries, modelOrder);

        System.out.println(model.aic()); // Get and display the model AIC
        System.out.println(model.coefficients()); // Get and display the estimated coefficients
        System.out.println(java.util.Arrays.toString(model.stdErrors()));

        plot(model.predictionErrors());

        Forecast forecast = model.forecast(12); // To specify the alpha significance level, add it as a second argument.

        System.out.println(forecast);

    }
}
