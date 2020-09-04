package ai.certifai.training.regression.powerprediction;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class PowerRegressionModel {

    private static Logger log = LoggerFactory.getLogger(PowerRegressionModel.class);

    public static void main(String[] args) throws IOException, InterruptedException {
        /*
         *  We would be using the Combined Cycle Power Plant to be our regression example.
         *  This dataset is obtained from https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip
         *  The description of the attributes:
         *  - Temperature (T) in the range 1.81°C and 37.11°C,
         *  - Ambient Pressure (AP) in the range 992.89-1033.30 milibar,
         *  - Relative Humidity (RH) in the range 25.56% to 100.16%
         *  - Exhaust Vacuum (V) in teh range 25.36-81.56 cm Hg
         *  - Net hourly electrical energy output (EP) 420.26-495.76 MW
         *  * Note that all the values are float/double values
         *
         *  TASK:
         *  ------
         *  1. Load the dataset using record reader
         *  2. Create schema based on the description
         *  3. Filling up the parameters for DataSetIterator for regression
         *  4. Splitting the data into training and test set with the ratio of 7:3
         *  5. Build the neural network with 3 hidden layer
         *  6. Fit your model with training set
         *  7. Evaluate your trained model with the test set
         *
         *  Good luck.
         *
         * */
        final int seed = 12345;
        final double learningRate = 0.001;
        final int nEpochs = 120;
        final int batchSize = 50;
//      importing the dataset
//        String path = new ClassPathResource("/power/power.csv").getFile().getAbsolutePath();
//        File file = new File(path);
//        /*      Use record reader and file split for data loading
//         *      Approximate around 2 lines of codes
//         * */

//      declaring schema of the data
//        Schema InputDataSchema = new Schema.Builder()
//                /*
//                 *
//                 * ENTER YOUR CODE HERE
//                 *
//                 * */   .build();
//        System.out.println("Initial Schema: " + InputDataSchema);

//        TransformProcess tp = new TransformProcess.Builder(InputDataSchema).build();
//      //adding the original data to a list for later transform purpose
//        List<List<Writable>> originalData = new ArrayList<>();
//        while (rr.hasNext()) {
//            List<Writable> data = rr.next();
//            originalData.add(data);
//        }

//        // transform data into final schema
//        List<List<Writable>> transformedData = LocalTransformExecutor.execute(originalData, tp);

//        //  Preparing to split the dataset into training set and test set
//        CollectionRecordReader collectionRecordReader = new CollectionRecordReader(transformedData);
//        DataSetIterator iterator = new RecordReaderDataSetIterator(//Your code here);
//
//        DataSet dataSet = iterator.next();
//        dataSet.shuffle();
//
//        SplitTestAndTrain testAndTrain = dataSet.splitTestAndTrain(0.7);
//
//        DataSet train = testAndTrain.getTrain();
//        DataSet test = testAndTrain.getTest();
//
//        INDArray features = train.getFeatures();
//        System.out.println("\nFeature shape: " + features.shapeInfoToString() + "\n");
//
//        //  Assigning dataset iterator for training purpose
//        ViewIterator trainIter = new ViewIterator(train, batchSize);
//        ViewIterator testIter = new ViewIterator(test, batchSize);
//      NN initialization
//      MultiLayerNetwork model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                /*
//                 *
//                 * ENTER YOUR CODE HERE
//                 *
//                 * */
//        model.init();
//        log.info("****************************************** UI SERVER **********************************************");
//        UIServer uiServer = UIServer.getInstance();
//        StatsStorage statsStorage = new InMemoryStatsStorage();
//        uiServer.attach(statsStorage);
//        model.setListeners(new ScoreIterationListener(10), new StatsListener(statsStorage));
//
//        // Model training - fit trainIter into model and evaluate model with testIter for each of nEpoch
//        log.info("\n*************************************** TRAINING **********************************************\n");
//
//        long timeX = System.currentTimeMillis();
////        fitting the model for nEpochs
//        for (int i = 0; i < nEpochs; i++) {
//            long time = System.currentTimeMillis();
//            trainIter.reset();
//            log.info("Epoch " + i);
//            model.fit(trainIter);
//            time = System.currentTimeMillis() - time;
//            log.info("************************** Done an epoch, TIME TAKEN: " + time + "ms **************************");
//
//            log.info("********************************** VALIDATING *************************************************");
//            RegressionEvaluation evaluation = model.evaluateRegression(testIter);
//            System.out.println(evaluation.stats());
//        }
//        long timeY = System.currentTimeMillis();
//        log.info("\n******************** TOTAL TIME TAKEN: " + (timeY - timeX) + "ms ******************************\n");
//
//        // Print out target values and predicted values
//        log.info("\n*************************************** PREDICTION **********************************************");
//
//        testIter.reset();
//      uncomment for predicted versus ground truth comparison
//        INDArray targetLabels = test.getLabels();
//        System.out.println("\nTarget shape: " + targetLabels.shapeInfoToString());
//
//        INDArray predictions = model.output(testIter);
//        System.out.println("\nPredictions shape: " + predictions.shapeInfoToString() + "\n");
//
//        System.out.println("Target \t\t\t Predicted");
//
//
//        for (int i = 0; i < targetLabels.rows(); i++) {
//            System.out.println(targetLabels.getRow(i) + "\t\t" + predictions.getRow(i));
//        }
//
//        // Plot the target values and predicted values
//        PlotUtil.visualizeRegression(targetLabels, predictions);
//
//        // Print out model summary
//        log.info("\n************************************* MODEL SUMMARY *******************************************");
//        System.out.println(model.summary());
    }
}
