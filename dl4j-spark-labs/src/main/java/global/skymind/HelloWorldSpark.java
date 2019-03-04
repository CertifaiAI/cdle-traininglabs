package global.skymind;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.solvers.accumulation.encoding.threshold.AdaptiveThresholdAlgorithm;
import org.deeplearning4j.spark.api.RDDTrainingApproach;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingMaster;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.v2.enums.MeshBuildMode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * This example reads in mnist data, perform training using Spark, and output the data.
 *
 * Set up a simple spark application up and runnning with two methods below
 * Declare useSparkLocal=true for local mode or declare masterIP for standalone
 *
 *      a. as a local mode (only provide flag useSparkLocal)
 *
 *      Step 1: mvn clean package
 *      Step 2: /usr/local/spark/bin/spark-submit --class org.deeplearning4j.HelloWorldSpark \
 *              path/to/helloworldspark-bin.jar \
 *              --useSparkLocal true \
 *
 *      b. as a standalone cluster (provide flag masterIP) - run this on a cluster of nodes.
 *
 *      Step 1: Start a Spark master node:
 *
 *          /usr/local/spark/sbin/start-master.sh
 *
 *      Step 2: Start Spark worker node/nodes:
 *
 *          /usr/local/spark/sbin/start-slave.sh spark://yourhostname.local:7077
 *
 *      Step 3: mvn clean package
 *
 *      Step 4: /usr/local/spark/bin/spark-submit --masterIP 192.168.1.6 \
 *                     --class org.deeplearning4j.HelloWorldSpark \
 *                     --numWorkersPerNode 1
 *                     path/to/helloworldspark-bin.jar
 *
 * Note: make sure (Spark 1 vs. Spark 2) and the Scala version (2.10 vs. 2.11)
 *       in pom.xml matches the cluster version to prevent runtime error
 *
 */
public class HelloWorldSpark {

    private static final Logger log = LoggerFactory.getLogger(HelloWorldSpark.class);

    @Parameter(names = "--useSparkLocal", description = "Use spark local (helper for testing/running without spark submit)", required = false, arity = 1)
    private boolean useSparkLocal = false;

    @Parameter(names = {"--masterIP"}, description = "Controller/master IP address - required. For example, 10.0.2.4", required = false)
    private String masterIP; //Example: 172.16.23.203

    @Parameter(names = {"--networkMask"}, description = "Network mask for Spark communication. For example, 10.0.0.0/16", required = false)
    private String networkMask; //Example: 172.16.0.0/16

    @Parameter(names = "--batchSizePerWorker", description = "Number of examples to fit each worker with", required = false)
    private int batchSizePerWorker = 15;

    @Parameter(names = {"--port"}, description = "Port number for Spark nodes. This can be any free port (port must be free on all nodes)", required = false)
    private int port = 40123;

    @Parameter(names = {"--gradientThreshold"}, description = "Gradient threshold", required = false)
    private double gradientThreshold = 1E-3;

    @Parameter(names = "--numEpochs", description = "Number of epochs for training", required = false)
    private int numEpochs = 2;

    @Parameter(names = {"--numWorkersPerNode"}, description = "Number of workers per Spark node. Usually use 1 per GPU, or 1 for CPU-only workers")
    private int numWorkersPerNode = 1;

    public static void main(String[] args) throws Exception
    {
        new HelloWorldSpark().entryPoint(args);
    }

    protected void entryPoint(String[] args) throws Exception
    {
        //Handle command line arguments
        JCommander jcmdr = new JCommander(this);
        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            //User provides invalid input -> print the usage info
            jcmdr.usage();
            try { Thread.sleep(500); } catch (Exception e2) { }
            throw e;
        }

        SparkConf sparkConf = new SparkConf();
        if (useSparkLocal)
        {
            log.info("Running local mode");
            sparkConf.setMaster("local[*]");
            masterIP = "127.0.0.1";
            networkMask = "127.0.0.0/16";
        }
        else
        {

            log.info("Running standalone mode");
            try
            {
                if(masterIP.isEmpty()) throw new Exception();

            }
            catch(Exception e)
            {
                log.error("Master IP not provided for standalone mode. Abort");
            }

            //badly written. Need to change to detect two . infront and replace with .0.0/16 in the end.
            int endIndex = masterIP.length() - 4;
            networkMask = masterIP.substring(0, endIndex) + ".0.0/16";
        }

        sparkConf.setAppName("DL4J Spark Mnist Example");

        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        //Load the data into memory then parallelize
        //Not commonly done - but is simple to use for this example
        DataSetIterator iterTrain = new MnistDataSetIterator(batchSizePerWorker, 10000, false, true, true, (long) 12345);
        DataSetIterator iterTest = new MnistDataSetIterator(batchSizePerWorker, 1000, false, true, false, (long) 12345);

        List<DataSet> trainDataList = new ArrayList<>();
        List<DataSet> testDataList = new ArrayList<>();

        while (iterTrain.hasNext()) {
            trainDataList.add(iterTrain.next());
        }
        while (iterTest.hasNext()) {
            testDataList.add(iterTest.next());
        }

        JavaRDD<DataSet> trainData = sc.parallelize(trainDataList);
        JavaRDD<DataSet> testData = sc.parallelize(testDataList);

        //----------------------------------
        //Create network configuration and conduct network training
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .activation(Activation.LEAKYRELU)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.01, 0.02))
                .l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(28 * 28).nOut(50).build())
                .layer(1, new DenseLayer.Builder().nIn(50).nOut(50).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(50).nOut(10).build())
                .build();

        //Configuration for Spark training
        VoidConfiguration voidConfiguration = VoidConfiguration.builder()
                //This can be any port, but it should be open for IN/OUT comms on all Spark node
                .unicastPort(port)
                //if you're running this example on Hadoop/YARN, please provide proper netmask for out-of-spark comms
                .networkMask(networkMask)
                //However, if you're running this example on Spark standalone cluster, you can rely on Spark internal addressing via $SPARK_PUBLIC_DNS env variables announced on each node
                .controllerAddress(useSparkLocal ? masterIP : null)
                //for < 32 nodes use PLAIN, else use MESH
                .meshBuildMode(MeshBuildMode.PLAIN)
                .build();


        TrainingMaster tm = new SharedTrainingMaster.Builder(voidConfiguration, batchSizePerWorker)
                .rngSeed(12345)
                .collectTrainingStats(false)
                .batchSizePerWorker(batchSizePerWorker)
                .thresholdAlgorithm(new AdaptiveThresholdAlgorithm(this.gradientThreshold))
                .workersPerNode(numWorkersPerNode)
                .build();


        //Create the Spark network
        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, conf, tm);


        //Execute training
        for (int i = 0; i < numEpochs; i++)
        {
            log.info("Completed Epoch {}", i);
            sparkNet.fit(trainData);

            //pause for slower execution. Refer to localhost:4040 for spark web ui
            TimeUnit.SECONDS.sleep(15);
        }


        //Perform evaluation (distributed)
        log.info("Evaluation");
        Evaluation evaluation = sparkNet.doEvaluation(testData, batchSizePerWorker, new Evaluation(10))[0]; //Work-around for 0.9.1 bug: see https://deeplearning4j.org/releasenotes
        log.info(evaluation.stats());

        //Delete the temp training files, now that we are done with them
        tm.deleteTempFiles(sc);

        log.info("***** Example Complete *****");
    }
}
