package global.skymind.solution.segmentation.imageUtils;

import org.datavec.image.loader.Java2DNativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class visualisation {

    private static Logger log = LoggerFactory.getLogger(visualisation.class);
    private static int displayWidth = 128;
    private static int displayHeight = 128;
    private static Java2DNativeImageLoader _Java2DNativeImageLoader;
    private static final String OUTPUT_PATH = "/out/";


    public static JFrame initFrame(String title) {
        JFrame frame = new JFrame();
        frame.setTitle(title);
//        frame.setTitle("Viz");
        frame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        frame.setLayout(new BorderLayout());
        return frame;
    }

    public static JPanel initPanel(JFrame frame, int displaySamples, int inputHeight, int inputWidth, int channels) {
        _Java2DNativeImageLoader = new Java2DNativeImageLoader(inputHeight, inputWidth, channels);

        JPanel panel = new JPanel();
        panel.setLayout(new GridLayout(displaySamples, 3, 1, 1));
        frame.add(panel, BorderLayout.CENTER);
        frame.setVisible(true);
        return panel;
    }

    public static void visualize(INDArray image, INDArray label, INDArray predict, JFrame frame, JPanel panel, int displaySamples, int outputHeight, int outputWidth) {
        panel.removeAll();
        for(int i=0; i<displaySamples; i++){
//            log.info(image.getRow(i).minNumber() + " - " + image.getRow(i).maxNumber());
            if(i<image.size(0)) {
                panel.add(getImage(image));
                panel.add(getImage(label));
//                panel.add(getBinaryImage(label, outputHeight, outputWidth));
//                panel.add(get255Image(predict.slice(i,0), outputHeight, outputWidth));
                panel.add(getImage(predict));
            }

        }
        frame.revalidate();
        frame.pack();
    }

    private static JLabel getImage(INDArray tensor) {
        BufferedImage bi = _Java2DNativeImageLoader.asBufferedImage(tensor.mul(255));
        ImageIcon orig = new ImageIcon(bi);
        Image imageScaled = orig.getImage().getScaledInstance((1 * displayWidth), (1 * displayHeight), Image.SCALE_REPLICATE);
        ImageIcon scaled = new ImageIcon(imageScaled);
        return new JLabel(scaled);
    }

    private static JLabel getBinaryImage(INDArray tensor, int w, int h) {
//        log.info(tensor.minNumber() + " to " + tensor.maxNumber());
        BufferedImage bi = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);
        for (int i = 0; i < w*h; i++) {
            int pixel = (int) (tensor.getDouble(i) * 255);
            bi.getRaster().setSample(i % w, i / h, 0, pixel);
        }
        ImageIcon orig = new ImageIcon(bi);
        Image imageScaled = orig.getImage().getScaledInstance((1 * displayWidth), (1 * displayHeight), Image.SCALE_REPLICATE);
        ImageIcon scaled = new ImageIcon(imageScaled);
        return new JLabel(scaled);
    }

    private static JLabel get255Image(INDArray tensor, int w, int h) {
//        log.info(tensor.minNumber() + " to " + tensor.maxNumber());
        BufferedImage bi = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);
        for (int i = 0; i < w*h; i++) {
            int pixel = (int) (tensor.getDouble(i) * 255);
            bi.getRaster().setSample(i % w, i / h, 0, pixel);
        }
        ImageIcon orig = new ImageIcon(bi);
        Image imageScaled = orig.getImage().getScaledInstance((1 * displayWidth), (1 * displayHeight), Image.SCALE_REPLICATE);
        ImageIcon scaled = new ImageIcon(imageScaled);
        return new JLabel(scaled);
    }

    private static BufferedImage INDArraytoBufferedImage(INDArray indarray, int outputHeight, int outputWidth) {
        BufferedImage bi = new BufferedImage(outputWidth, outputHeight, BufferedImage.TYPE_3BYTE_BGR);

        for (int i = 0; i < outputWidth * outputHeight; i++) {
            int rgb = (int)indarray.getFloat(i);
            bi.setRGB(i % outputWidth, i / outputHeight, rgb);
        }
        return bi;
    }

    public static void export(File dir, INDArray image, INDArray label, INDArray predict, int count) throws IOException {
        for (int i = 0; i < image.size(0); i++) {
            BufferedImage oriImage = _Java2DNativeImageLoader.asBufferedImage(image.slice(i, 0).mul(255));
            BufferedImage labelImage = _Java2DNativeImageLoader.asBufferedImage(label.slice(i, 0).mul(255));
            BufferedImage predictImage = _Java2DNativeImageLoader.asBufferedImage(predict.slice(i, 0).mul(255));
//            INDArrayIndex[] imageMat = {NDArrayIndex.indices(i), NDArrayIndex.indices(1), NDArrayIndex.all(), NDArrayIndex.all()};
//            BufferedImage labelImage = INDArraytoBufferedImage(Nd4j.argMax(label.get(imageMat), 1), displayHeight, displayWidth);
//            BufferedImage predictImage = INDArraytoBufferedImage(Nd4j.argMax(predict.get(imageMat), 1), displayHeight, displayWidth);
            ImageIO.write(oriImage, "png", new File(dir.getAbsolutePath() + "/" + (i + count) + "_image.png"));
            ImageIO.write(labelImage, "png", new File(dir.getAbsolutePath() + "/" + (i + count) + "_label.png"));
            ImageIO.write(predictImage, "png", new File(dir.getAbsolutePath() +"/" + (i + count) + "_predict.png"));
        }
    }


}
