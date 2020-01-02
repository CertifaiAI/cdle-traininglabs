package global.skymind.training.segmentation.car;

import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URI;

public class CustomLabelGenerator implements PathLabelGenerator {
    private static Logger log = LoggerFactory.getLogger(CustomLabelGenerator.class);

    private final int height;
    private final int width;
    private final int channels;
    private final NativeImageLoader imageLoader;

    /*
     * Instructions for this lab exercise:
     * STEP 1: Complete the code in order to find corresponding labels for the training images.
     *
     * */

    @Override
    public boolean inferLabelClasses() {
        return false;
    }

    public CustomLabelGenerator(int height, int width, int channels){
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.imageLoader = new NativeImageLoader(this.height, this.width, this.channels);
    }

    // STEP 1: Write your code here to generate labels for all images.
    @Override
    public Writable getLabelForPath(String path) {


            /**
             * ENTER YOUR CODE HERE
             * **/

        return null;
    }

    @Override
    public Writable getLabelForPath(URI uri) {
        return this.getLabelForPath(uri.getPath());
    }




}
