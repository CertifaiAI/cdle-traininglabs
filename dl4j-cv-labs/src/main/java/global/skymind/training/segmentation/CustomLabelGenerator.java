package global.skymind.training.segmentation;

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

    @Override
    public Writable getLabelForPath(String path) {
        try
        {
            String labelPath = path.replace("\\inputs\\", "\\masks\\");

            NDArrayWritable label = new NDArrayWritable(imageLoader.asMatrix(new File(labelPath)) );

//            System.out.println(labelPath);
//            System.out.println(label);

            INDArray labelINDArray = label.get();

//            // adding 255 to green channel, so that 0-1 normalisation would result in one-hot encoding for 3 classes.
//            INDArray newValue = labelINDArray.get(NDArrayIndex.all(), NDArrayIndex.point(1),  NDArrayIndex.all()).add(labelINDArray.sum(1)).sub(255).mul(-1) ;
//            labelINDArray.get(NDArrayIndex.all(), NDArrayIndex.point(1),  NDArrayIndex.all()).assign(newValue);
//
//            // normalise to 0-1 scale
            label.set( labelINDArray.div(255));

            return label ;

        }
        catch (IOException ioe)
        {
            ioe.printStackTrace();
            return null;
        }
    }

    @Override
    public Writable getLabelForPath(URI uri) {
        return this.getLabelForPath(uri.getPath());
    }


    public INDArray getLabelForPathNDArray(URI uri) {
        try
        {
            String labelURI = uri.toString().replace("\\inputs\\", "\\masks\\");

            return  imageLoader.asMatrix(new File(URI.create(labelURI))).div(255);
//            return  imageLoader.asMatrix(new File(URI.create(labelURI)));

        }
        catch (IOException ioe)
        {
            ioe.printStackTrace();
            return null;
        }
    }

}
