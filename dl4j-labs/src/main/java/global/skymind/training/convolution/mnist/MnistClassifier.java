/*
 *
 *  * ******************************************************************************
 *  *  * Copyright (c) 2019 Skymind AI Bhd.
 *  *  * Copyright (c) 2020 CertifAI Sdn. Bhd.
 *  *  *
 *  *  * This program and the accompanying materials are made available under the
 *  *  * terms of the Apache License, Version 2.0 which is available at
 *  *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *  *
 *  *  * Unless required by applicable law or agreed to in writing, software
 *  *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  *  * License for the specific language governing permissions and limitations
 *  *  * under the License.
 *  *  *
 *  *  * SPDX-License-Identifier: Apache-2.0
 *  *  *****************************************************************************
 *
 *
 */

package global.skymind.training.convolution.mnist;

import javafx.application.Application;
import javafx.embed.swing.SwingFXUtils;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Label;
import javafx.scene.image.ImageView;
import javafx.scene.image.WritableImage;
import javafx.scene.input.KeyCode;
import javafx.scene.input.MouseButton;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.shape.StrokeLineCap;
import javafx.stage.Stage;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;

/**
 * Hello World with Mnist DataSet
 *
 ******************************************************
 * Look for LAB STEP below. Uncomment to proceed.
 *
 * 1. Set up Model configuration
 * 2. Test on a single image
 * *****************************************************
 */
public class MnistClassifier extends Application {

    private static final Logger log = LoggerFactory.getLogger(MnistClassifier.class);
    private static final int canvasWidth = 150;
    private static final int canvasHeight = 150;

    private static final int height = 28;
    private static final int width = 28;
    private static final int channels = 1; // single channel for grayscale images
    private static final int outputNum = 10; // 10 digits classification
    private static final int batchSize = 54;
    private static final int nEpochs = 1;
    private static final double learningRate = 0.001;
    private static MultiLayerNetwork model = null;

    private static final int seed = 1234;

    public static void main(String[] args) throws Exception
    {
    /*
    Create an iterator using the batch size for one iteration
    */
        log.info("Data load and vectorization...");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize,true, seed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize,false, seed);


    /*
   #### LAB STEP 1 #####
   Model configuration
    */
        log.info("Network configuration and training...");

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Nesterovs(learningRate, Nesterovs.DEFAULT_NESTEROV_MOMENTUM))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(50).build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(height, width, 1)) // InputType.convolutional for normal image
                .backpropType(BackpropType.Standard)
                .build();

        model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        // evaluation while training (the score should go down)
        for (int i = 0; i < nEpochs; i++) {
            model.fit(mnistTrain);

            log.info("Completed epoch {}", i);
            Evaluation eval = model.evaluate(mnistTest);
            log.info(eval.stats());
            mnistTrain.reset();
            mnistTest.reset();
        }

    /*
   #### LAB STEP 2 #####
   Test on a single image
    */

        launch();
    }

    @Override
    public void start(Stage stage) throws Exception {
        Canvas canvas = new Canvas(canvasWidth, canvasHeight);
        GraphicsContext ctx = canvas.getGraphicsContext2D();

        ImageView imgView = new ImageView();
        imgView.setFitHeight(100);
        imgView.setFitWidth(100);
        ctx.setLineWidth(10);
        ctx.setLineCap(StrokeLineCap.SQUARE);
        Label lblResult = new Label();

        HBox hbBottom = new HBox(10, imgView, lblResult);
        hbBottom.setAlignment(Pos.CENTER);
        VBox root = new VBox(5, canvas, hbBottom);
        root.setAlignment(Pos.CENTER);

        Scene scene = new Scene(root, 680, 300);
        stage.setScene(scene);
        stage.setTitle("Draw a digit and hit enter (right-click to clear)");
        stage.setResizable(false);
        stage.show();

        canvas.setOnMousePressed(e -> {
            ctx.setStroke(Color.WHITE);
            ctx.beginPath();
            ctx.moveTo(e.getX(), e.getY());
            ctx.stroke();
        });
        canvas.setOnMouseDragged(e -> {
            ctx.setStroke(Color.WHITE);
            ctx.lineTo(e.getX(), e.getY());
            ctx.stroke();
        });
        canvas.setOnMouseClicked(e -> {
            if (e.getButton() == MouseButton.SECONDARY) {
                clear(ctx);
            }
        });
        canvas.setOnKeyReleased(e -> {
            if (e.getCode() == KeyCode.ENTER) {
                BufferedImage scaledImg = getScaledImage(canvas);
                imgView.setImage(SwingFXUtils.toFXImage(scaledImg, null));
                try {
                    predictImage(scaledImg, lblResult);
                } catch (Exception e1) {
                    e1.printStackTrace();
                }
            }
        });
        clear(ctx);
        canvas.requestFocus();
    }

    private void clear(GraphicsContext ctx) {
        ctx.setFill(Color.BLACK);
        ctx.fillRect(0, 0, 300, 300);
    }

    private BufferedImage getScaledImage(Canvas canvas) {
        WritableImage writableImage = new WritableImage(canvasWidth, canvasHeight);
        canvas.snapshot(null, writableImage);
        Image tmp = SwingFXUtils.fromFXImage(writableImage, null).getScaledInstance(28, 28, Image.SCALE_SMOOTH);
        BufferedImage scaledImg = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        Graphics graphics = scaledImg.getGraphics();
        graphics.drawImage(tmp, 0, 0, null);
        graphics.dispose();
        return scaledImg;
    }

    private void predictImage(BufferedImage img, Label lbl) throws IOException {
        NativeImageLoader loader = new NativeImageLoader(28, 28, 1, true);
        INDArray image = loader.asRowVector(img).reshape(new int[]{1,784});
        //ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        //scaler.transform(image);
        INDArray output = model.output(image);
        lbl.setText("Prediction: " + model.predict(image)[0] + "\n " + output);
    }

}
