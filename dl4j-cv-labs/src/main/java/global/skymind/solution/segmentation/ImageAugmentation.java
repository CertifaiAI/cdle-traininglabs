package global.skymind.solution.segmentation;

import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.datavec.image.data.ImageWritable;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.datavec.image.transform.RotateImageTransform;
import org.nd4j.linalg.primitives.Pair;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class ImageAugmentation {
    protected static final long seed = 12345;
    private static final Random random = new Random(seed);

    public static void main(String[] args) throws IOException{
        File imagesPath = new File(System.getProperty("user.home"), ".deeplearning4j/data/data-science-bowl-2018/data-science-bowl-2018/data-science-bowl-2018-2/train/inputs");
        File[] files = imagesPath.listFiles();

        ImageTransform flip = new FlipImageTransform();
        ImageTransform rotate = new RotateImageTransform(random, 30);

        List<Pair<ImageTransform, Double>> listOfTransform = Arrays.asList(
                new Pair<>(rotate, 1.0),
                new Pair<>(flip, 0.7)
        );

        PipelineImageTransform transformPipeline = new PipelineImageTransform(listOfTransform, false);


        NativeImageLoader niLoader= new NativeImageLoader(224,224,1,flip);

        File augmentedImgPath = new File(System.getProperty("user.home"),".deeplearning4j/data/data-science-bowl-2018/data-science-bowl-2018/data-science-bowl-2018-2/train/augmented_inputs");

        if (!augmentedImgPath.exists() ) {
            augmentedImgPath.mkdir();
        }


        if (files != null) {
            for (File f : files){

                // ImageWritable -> Frame -> BufferedImage -> png
                ImageWritable iw = niLoader.asWritable(f);
//                ImageWritable transformed = flip.transform(iw);
                ImageWritable transformed = transformPipeline.transform(iw);


                Frame frame = transformed.getFrame() ;
                Java2DFrameConverter converter = new Java2DFrameConverter();
                BufferedImage bimage = converter.convert(frame);

                ImageIO.write(bimage, "jpg", new File(augmentedImgPath + "/" + f.getName() ));
            }
        }


    }

}


