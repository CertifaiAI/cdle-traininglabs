package pro.skymind.ts.decomposition;

import com.github.servicenow.ds.stats.stl.SeasonalTrendLoess;
import com.github.signaflo.timeseries.TestData;

import java.util.Arrays;

public class servicenowSTLDecompExample {
    public static void main(String[] args) {
        double[] values = TestData.debitcards.asArray(); // Monthly time-series data

        SeasonalTrendLoess.Builder builder = new SeasonalTrendLoess.Builder();
        SeasonalTrendLoess smoother = builder.
                setPeriodLength(12).    // Data has a period of 12
                setSeasonalWidth(35).   // Monthly data smoothed over 35 years
                setNonRobust().         // Not expecting outliers, so no robustness iterations
                buildSmoother(values);

        SeasonalTrendLoess.Decomposition stl = smoother.decompose();
        double[] seasonal = stl.getSeasonal();
        double[] trend = stl.getTrend();
        double[] residual = stl.getResidual();

        Arrays.stream(seasonal).forEach(num -> System.out.print(num + ","));
        System.out.println();
        Arrays.stream(trend).forEach(num -> System.out.print(num + ","));
        System.out.println();
        Arrays.stream(residual).forEach(num -> System.out.print(num + ","));
    }
}
