package global.skymind.solution.segmentation;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Random;

public class Inference {
    private static final Logger log = LoggerFactory.getLogger(Inference.class);
    private static File modelFilename = new File("segmentUNetFineTune.zip");
    private static final int height = 224;
    private static final int width = 224;
    private static final int channels = 1;
    protected static final long seed = 12345;
    private static final Random random = new Random(seed);

    public static void main(String[] args) throws Exception {

        if (modelFilename.exists()) {
            log.info("Load model...");
            ComputationGraph model = ModelSerializer.restoreComputationGraph(modelFilename);
        } else {
            log.info("Model not found.");
        }

//        File imagesPath = new File(System.getProperty("user.home"), ".deeplearning4j/data/data-science-bowl-2018/data-science-bowl-2018/inputs_no_alpha");
//        System.out.println(imagesPath);
//        FileSplit imageSplit = new FileSplit(imagesPath, NativeImageLoader.ALLOWED_FORMATS, random);
//
//        //Load labels
//        CustomLabelGenerator labelMaker = new CustomLabelGenerator(height, width, 1); // labels have 1 channel
//        BalancedPathFilter imageSplitPathFilter = new BalancedPathFilter(random, NativeImageLoader.ALLOWED_FORMATS, labelMaker);


    }

}