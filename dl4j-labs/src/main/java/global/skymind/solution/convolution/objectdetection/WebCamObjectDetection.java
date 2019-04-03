package global.skymind.solution.convolution.objectdetection;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ColorConversionTransform;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.deeplearning4j.zoo.util.darknet.VOCLabels;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.event.KeyEvent;
import java.util.List;

import static org.bytedeco.javacpp.opencv_core.FONT_HERSHEY_DUPLEX;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.opencv.imgproc.Imgproc.COLOR_BGR2RGB;

/**
 * YOLO detection with camera
 * Note 1: Swap between camera by using createDefault parameter
 * Note 2: flip the camera if opening front camera
 */

public class WebCamObjectDetection
{
    private static Thread thread;


    private static final int gridWidth = 13;
    private static final int gridHeight = 13;

    private static double detectionThreshold = 0.5;
    private static final int tinyyolowidth = 416;
    private static final int tinyyoloheight = 416;


    public static void main(String[] args) throws Exception {

        //swap between camera with 0 -? on the parameter
        FrameGrabber grabber = FrameGrabber.createDefault(0);
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

        grabber.start();

        String winName = "Object Detection";
        CanvasFrame canvas = new CanvasFrame(winName);

        int w = grabber.getImageWidth();
        int h = grabber.getImageHeight();


        canvas.setCanvasSize(w, h);

        ZooModel model = TinyYOLO.builder().numClasses(0).build();
        ComputationGraph initializedModel = (ComputationGraph) model.initPretrained();


        NativeImageLoader loader = new NativeImageLoader(tinyyolowidth, tinyyoloheight, 3, new ColorConversionTransform(COLOR_BGR2RGB));
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        VOCLabels labels = new VOCLabels();

        System.out.println("Start running video");

        while (true)
        {
            Frame frame = grabber.grab();

            //if a thread is null, create new thread
            if (thread == null)
            {
                thread = new Thread(() ->
                {
                    while (frame != null)
                    {
                        try
                        {
                            opencv_core.Mat rawImage = converter.convert(frame);
                            //opencv_core.Mat rawImage = new opencv_core.Mat();
                            //Flip the camera if opening front camera
                            //opencv_core.flip(inputMat, rawImage, 1);

                            opencv_core.Mat resizeImage = new opencv_core.Mat();
                            resize(rawImage, resizeImage, new opencv_core.Size(tinyyolowidth, tinyyoloheight));

                            INDArray inputImage = loader.asMatrix(resizeImage);
                            scaler.transform(inputImage);
                            INDArray outputs = initializedModel.outputSingle(inputImage);
                            List<DetectedObject> objs = YoloUtils.getPredictedObjects(Nd4j.create(((TinyYOLO) model).getPriorBoxes()), outputs, detectionThreshold, 0.4);


                            for (DetectedObject obj : objs) {
                                double[] xy1 = obj.getTopLeftXY();
                                double[] xy2 = obj.getBottomRightXY();
                                String label = labels.getLabel(obj.getPredictedClass());
                                int x1 = (int) Math.round(w * xy1[0] / gridWidth);
                                int y1 = (int) Math.round(h * xy1[1] / gridHeight);
                                int x2 = (int) Math.round(w * xy2[0] / gridWidth);
                                int y2 = (int) Math.round(h * xy2[1] / gridHeight);
                                rectangle(rawImage, new opencv_core.Point(x1, y1), new opencv_core.Point(x2, y2), opencv_core.Scalar.RED, 2, 0, 0);
                                putText(rawImage, label, new opencv_core.Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, opencv_core.Scalar.GREEN);
                            }

                            canvas.showImage(converter.convert(rawImage));

                        }
                        catch (Exception e)
                        {
                            throw new RuntimeException(e);
                        }
                    }
                });
                thread.start();
            }


            KeyEvent t = canvas.waitKey(33);

            if ((t != null) && (t.getKeyCode() == KeyEvent.VK_Q)) {
                break;
            }
        }

        canvas.dispose();

    }

}