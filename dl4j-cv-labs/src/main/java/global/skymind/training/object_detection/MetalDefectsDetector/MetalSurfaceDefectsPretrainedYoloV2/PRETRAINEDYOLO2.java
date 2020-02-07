package global.skymind.training.object_detection.MetalDefectsDetector.MetalSurfaceDefectsPretrainedYoloV2;

import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.Builder;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.AlgoMode;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooType;
import org.deeplearning4j.zoo.model.helper.DarknetHelper;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;

import java.util.Arrays;

public class PRETRAINEDYOLO2 extends ZOOMODEL {
    public static final double[][] DEFAULT_PRIOR_BOXES = new double[][]{{0.57273D, 0.677385D}, {1.87446D, 2.06253D}, {3.33843D, 5.47434D}, {7.88282D, 3.52778D}, {9.77052D, 9.16828D}};
    private int nBoxes;
    private double[][] priorBoxes;
    private long seed;
    private int[] inputShape;
    private int numClasses;
    private IUpdater updater;
    private CacheMode cacheMode;
    private WorkspaceMode workspaceMode;
    private AlgoMode cudnnAlgoMode;

    private PRETRAINEDYOLO2() {
    }

    public String pretrainedUrl(PretrainedType pretrainedType) {
        return pretrainedType == PretrainedType.IMAGENET ? DL4JResources.getURLString("models/PRETRAINEDYOLO2_dl4j_inference.v3.zip") : null;
    }

    public long pretrainedChecksum(PretrainedType pretrainedType) {
        return pretrainedType == PretrainedType.IMAGENET ? 3658373840L : 0L;
    }

    public Class<? extends Model> modelType() {
        return ComputationGraph.class;
    }

    public ComputationGraphConfiguration conf() {
        INDArray priors = Nd4j.create(this.priorBoxes);
        GraphBuilder graphBuilder = (new Builder()).seed(this.seed).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).gradientNormalization(GradientNormalization.RenormalizeL2PerLayer).gradientNormalizationThreshold(1.0D).updater(this.updater).l2(1.0E-5D).activation(Activation.IDENTITY).cacheMode(this.cacheMode).trainingWorkspaceMode(this.workspaceMode).inferenceWorkspaceMode(this.workspaceMode).cudnnAlgoMode(this.cudnnAlgoMode).graphBuilder().addInputs(new String[]{"input"}).setInputTypes(new InputType[]{InputType.convolutional((long)this.inputShape[2], (long)this.inputShape[1], (long)this.inputShape[0])});
        DarknetHelper.addLayers(graphBuilder, 1, 3, this.inputShape[0], 32, 2);
        DarknetHelper.addLayers(graphBuilder, 2, 3, 32, 64, 2);
        DarknetHelper.addLayers(graphBuilder, 3, 3, 64, 128, 0);
        DarknetHelper.addLayers(graphBuilder, 4, 1, 128, 64, 0);
        DarknetHelper.addLayers(graphBuilder, 5, 3, 64, 128, 2);
        DarknetHelper.addLayers(graphBuilder, 6, 3, 128, 256, 0);
        DarknetHelper.addLayers(graphBuilder, 7, 1, 256, 128, 0);
        DarknetHelper.addLayers(graphBuilder, 8, 3, 128, 256, 2);
        DarknetHelper.addLayers(graphBuilder, 9, 3, 256, 512, 0);
        DarknetHelper.addLayers(graphBuilder, 10, 1, 512, 256, 0);
        DarknetHelper.addLayers(graphBuilder, 11, 3, 256, 512, 0);
        DarknetHelper.addLayers(graphBuilder, 12, 1, 512, 256, 0);
        DarknetHelper.addLayers(graphBuilder, 13, 3, 256, 512, 2);
        DarknetHelper.addLayers(graphBuilder, 14, 3, 512, 1024, 0);
        DarknetHelper.addLayers(graphBuilder, 15, 1, 1024, 512, 0);
        DarknetHelper.addLayers(graphBuilder, 16, 3, 512, 1024, 0);
        DarknetHelper.addLayers(graphBuilder, 17, 1, 1024, 512, 0);
        DarknetHelper.addLayers(graphBuilder, 18, 3, 512, 1024, 0);
        DarknetHelper.addLayers(graphBuilder, 19, 3, 1024, 1024, 0);
        DarknetHelper.addLayers(graphBuilder, 20, 3, 1024, 1024, 0);
        DarknetHelper.addLayers(graphBuilder, 21, "activation_13", 1, 512, 64, 0, 0);
        graphBuilder.addLayer("rearrange_21", (new org.deeplearning4j.nn.conf.layers.SpaceToDepthLayer.Builder(2)).build(), new String[]{"activation_21"}).addVertex("concatenate_21", new MergeVertex(), new String[]{"rearrange_21", "activation_20"});
        DarknetHelper.addLayers(graphBuilder, 22, "concatenate_21", 3, 1280, 1024, 0, 0);
        graphBuilder.addLayer("convolution2d_23", ((org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder)((org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder)((org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder)((org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder)((org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder)((org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder)((org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder)(new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(new int[]{1, 1})).nIn(1024)).nOut(this.nBoxes * (5 + this.numClasses))).weightInit(WeightInit.XAVIER)).stride(new int[]{1, 1}).convolutionMode(ConvolutionMode.Same)).weightInit(WeightInit.RELU)).activation(Activation.IDENTITY)).cudnnAlgoMode(this.cudnnAlgoMode)).build(), new String[]{"activation_22"}).addLayer("outputs", (new org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer.Builder()).boundingBoxPriors(priors).build(), new String[]{"convolution2d_23"}).setOutputs(new String[]{"outputs"});
        return graphBuilder.build();
    }

    public ComputationGraph init() {
        ComputationGraph model = new ComputationGraph(this.conf());
        model.init();
        return model;
    }

    public ModelMetaData metaData() {
        return new ModelMetaData(new int[][]{this.inputShape}, 1, ZooType.CNN);
    }

    public void setInputShape(int[][] inputShape) {
        this.inputShape = inputShape[0];
    }

    private static int $default$nBoxes() {
        return 5;
    }

    private static double[][] $default$priorBoxes() {
        return DEFAULT_PRIOR_BOXES;
    }

    private static long $default$seed() {
        return 1234L;
    }

    private static int[] $default$inputShape() {
        return new int[]{3, 608, 608};
    }

    private static int $default$numClasses() {
        return 0;
    }

    private static IUpdater $default$updater() {
        return new Adam(0.001D);
    }

    private static CacheMode $default$cacheMode() {
        return CacheMode.NONE;
    }

    private static WorkspaceMode $default$workspaceMode() {
        return WorkspaceMode.ENABLED;
    }

    private static AlgoMode $default$cudnnAlgoMode() {
        return AlgoMode.PREFER_FASTEST;
    }

    public static PRETRAINEDYOLO2.PRETRAINEDYOLO2Builder builder() {
        return new PRETRAINEDYOLO2.PRETRAINEDYOLO2Builder();
    }

    public PRETRAINEDYOLO2(int nBoxes, double[][] priorBoxes, long seed, int[] inputShape, int numClasses, IUpdater updater, CacheMode cacheMode, WorkspaceMode workspaceMode, AlgoMode cudnnAlgoMode) {
        this.nBoxes = nBoxes;
        this.priorBoxes = priorBoxes;
        this.seed = seed;
        this.inputShape = inputShape;
        this.numClasses = numClasses;
        this.updater = updater;
        this.cacheMode = cacheMode;
        this.workspaceMode = workspaceMode;
        this.cudnnAlgoMode = cudnnAlgoMode;
    }

    public int getNBoxes() {
        return this.nBoxes;
    }

    public double[][] getPriorBoxes() {
        return this.priorBoxes;
    }

    public static class PRETRAINEDYOLO2Builder {
        private boolean nBoxes$set;
        private int nBoxes;
        private boolean priorBoxes$set;
        private double[][] priorBoxes;
        private boolean seed$set;
        private long seed;
        private boolean inputShape$set;
        private int[] inputShape;
        private boolean numClasses$set;
        private int numClasses;
        private boolean updater$set;
        private IUpdater updater;
        private boolean cacheMode$set;
        private CacheMode cacheMode;
        private boolean workspaceMode$set;
        private WorkspaceMode workspaceMode;
        private boolean cudnnAlgoMode$set;
        private AlgoMode cudnnAlgoMode;

        PRETRAINEDYOLO2Builder() {
        }

        public PRETRAINEDYOLO2.PRETRAINEDYOLO2Builder nBoxes(int nBoxes) {
            this.nBoxes = nBoxes;
            this.nBoxes$set = true;
            return this;
        }

        public PRETRAINEDYOLO2.PRETRAINEDYOLO2Builder priorBoxes(double[][] priorBoxes) {
            this.priorBoxes = priorBoxes;
            this.priorBoxes$set = true;
            return this;
        }

        public PRETRAINEDYOLO2.PRETRAINEDYOLO2Builder seed(long seed) {
            this.seed = seed;
            this.seed$set = true;
            return this;
        }

        public PRETRAINEDYOLO2.PRETRAINEDYOLO2Builder inputShape(int[] inputShape) {
            this.inputShape = inputShape;
            this.inputShape$set = true;
            return this;
        }

        public PRETRAINEDYOLO2.PRETRAINEDYOLO2Builder numClasses(int numClasses) {
            this.numClasses = numClasses;
            this.numClasses$set = true;
            return this;
        }

        public PRETRAINEDYOLO2.PRETRAINEDYOLO2Builder updater(IUpdater updater) {
            this.updater = updater;
            this.updater$set = true;
            return this;
        }

        public PRETRAINEDYOLO2.PRETRAINEDYOLO2Builder cacheMode(CacheMode cacheMode) {
            this.cacheMode = cacheMode;
            this.cacheMode$set = true;
            return this;
        }

        public PRETRAINEDYOLO2.PRETRAINEDYOLO2Builder workspaceMode(WorkspaceMode workspaceMode) {
            this.workspaceMode = workspaceMode;
            this.workspaceMode$set = true;
            return this;
        }

        public PRETRAINEDYOLO2.PRETRAINEDYOLO2Builder cudnnAlgoMode(AlgoMode cudnnAlgoMode) {
            this.cudnnAlgoMode = cudnnAlgoMode;
            this.cudnnAlgoMode$set = true;
            return this;
        }

        public PRETRAINEDYOLO2 build() {
            int nBoxes = this.nBoxes;
            if (!this.nBoxes$set) {
                nBoxes = PRETRAINEDYOLO2.$default$nBoxes();
            }

            double[][] priorBoxes = this.priorBoxes;
            if (!this.priorBoxes$set) {
                priorBoxes = PRETRAINEDYOLO2.$default$priorBoxes();
            }

            long seed = this.seed;
            if (!this.seed$set) {
                seed = PRETRAINEDYOLO2.$default$seed();
            }

            int[] inputShape = this.inputShape;
            if (!this.inputShape$set) {
                inputShape = PRETRAINEDYOLO2.$default$inputShape();
            }

            int numClasses = this.numClasses;
            if (!this.numClasses$set) {
                numClasses = PRETRAINEDYOLO2.$default$numClasses();
            }

            IUpdater updater = this.updater;
            if (!this.updater$set) {
                updater = PRETRAINEDYOLO2.$default$updater();
            }

            CacheMode cacheMode = this.cacheMode;
            if (!this.cacheMode$set) {
                cacheMode = PRETRAINEDYOLO2.$default$cacheMode();
            }

            WorkspaceMode workspaceMode = this.workspaceMode;
            if (!this.workspaceMode$set) {
                workspaceMode = PRETRAINEDYOLO2.$default$workspaceMode();
            }

            AlgoMode cudnnAlgoMode = this.cudnnAlgoMode;
            if (!this.cudnnAlgoMode$set) {
                cudnnAlgoMode = PRETRAINEDYOLO2.$default$cudnnAlgoMode();
            }

            return new PRETRAINEDYOLO2(nBoxes, priorBoxes, seed, inputShape, numClasses, updater, cacheMode, workspaceMode, cudnnAlgoMode);
        }

        public String toString() {
            return "PRETRAINEDYOLO2.PRETRAINEDYOLO2Builder(nBoxes=" + this.nBoxes + ", priorBoxes=" + Arrays.deepToString(this.priorBoxes) + ", seed=" + this.seed + ", inputShape=" + Arrays.toString(this.inputShape) + ", numClasses=" + this.numClasses + ", updater=" + this.updater + ", cacheMode=" + this.cacheMode + ", workspaceMode=" + this.workspaceMode + ", cudnnAlgoMode=" + this.cudnnAlgoMode + ")";
        }
    }
}
