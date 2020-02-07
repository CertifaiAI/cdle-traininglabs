package global.skymind.solution.object_detection.MetalDefectsDetector.MetalSurfaceDefectsPretrainedYoloV2;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.zip.Adler32;
import java.util.zip.Checksum;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.common.resources.ResourceType;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.InstantiableModel;
import org.deeplearning4j.zoo.PretrainedType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class ZOOMODEL<T> implements InstantiableModel {
    private static final Logger log = LoggerFactory.getLogger(ZOOMODEL.class);

    public ZOOMODEL() {
    }

    public boolean pretrainedAvailable(PretrainedType pretrainedType) {
        return this.pretrainedUrl(pretrainedType) != null;
    }

    public Model initPretrained() throws IOException {
        return this.initPretrained(PretrainedType.IMAGENET);
    }

    public <M extends Model> M initPretrained(PretrainedType pretrainedType) throws IOException {
//        String remoteUrl = this.pretrainedUrl(pretrainedType);
        String remoteUrl = "https://archive.org/download/pretrainedmetalsurfacedefectsyolov2/Pretrained_MetalSurfaceDefects_yolov2.zip";
        if (remoteUrl == null) {
            throw new UnsupportedOperationException("Pretrained " + pretrainedType + " weights are not available for this model.");
        } else {
            String localFilename = (new File(remoteUrl)).getName();
            File rootCacheDir = DL4JResources.getDirectory(ResourceType.ZOO_MODEL, this.modelName());
            File cachedFile = new File(rootCacheDir, localFilename);
            if (!cachedFile.exists()) {
                log.info("Downloading model to " + cachedFile.toString());
                FileUtils.copyURLToFile(new URL(remoteUrl), cachedFile);
            } else {
                log.info("Using cached model at " + cachedFile.toString());
            }


            if (this.modelType() == MultiLayerNetwork.class) {
                return (M) ModelSerializer.restoreMultiLayerNetwork(cachedFile);
            } else if (this.modelType() == ComputationGraph.class) {
                return (M) ModelSerializer.restoreComputationGraph(cachedFile);
            } else {
                throw new UnsupportedOperationException("Pretrained models are only supported for MultiLayerNetwork and ComputationGraph.");
            }
        }
    }

    public String modelName() {
        return this.getClass().getSimpleName().toLowerCase();
    }
}
