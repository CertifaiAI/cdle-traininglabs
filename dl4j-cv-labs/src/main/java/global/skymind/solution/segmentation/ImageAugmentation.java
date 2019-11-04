package global.skymind.solution.segmentation;

import global.skymind.Helper;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.datavec.image.data.ImageWritable;
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
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class ImageAugmentation {
    protected static final long seed = 12345;
    private static final Random random = new Random(seed);
    private static String inputDir;

    public static void main(String[] args) throws IOException{
        /*
         * ONLY run to generate more samples
         *
         * */
        inputDir = Paths.get(
                System.getProperty("user.home"),
                Helper.getPropValues("dl4j_home.data")
        ).toString();

        File imagesPath = new File(Paths.get(inputDir, "data-science-bowl-2018","data-science-bowl-2018","data-science-bowl-2018-2","train","inputs").toString());
        File[] files = imagesPath.listFiles();

        ImageTransform flip = new FlipImageTransform();
        ImageTransform rotate = new RotateImageTransform(random, 30);

        List<Pair<ImageTransform, Double>> listOfTransform = Arrays.asList(
                new Pair<>(rotate, 1.0),
                new Pair<>(flip, 0.7)
        );

        PipelineImageTransform transformPipeline = new PipelineImageTransform(listOfTransform, false);

        NativeImageLoader niLoader= new NativeImageLoader(224,224,1,flip);

        File augmentedImgFolder = new File(Paths.get(inputDir, "data-science-bowl-2018","data-science-bowl-2018","data-science-bowl-2018-2","train","augmented_inputs").toString());

        if (!augmentedImgFolder.exists() ) {
            augmentedImgFolder.mkdir();
        }


        if (files != null) {
            for (File f : files){

                // ImageWritable -> Frame -> BufferedImage -> png
                ImageWritable iw = niLoader.asWritable(f);
                ImageWritable transformed = transformPipeline.transform(iw);


                Frame frame = transformed.getFrame() ;
                Java2DFrameConverter converter = new Java2DFrameConverter();
                BufferedImage bimage = converter.convert(frame);

                File augmentedImgPath = new File(Paths.get(inputDir, "data-science-bowl-2018","data-science-bowl-2018","data-science-bowl-2018-2","train","augmented_inputs", f.getName()).toString());
                ImageIO.write(bimage, "jpg", augmentedImgPath);
            }
        }


    }

}


