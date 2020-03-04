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

package global.skymind.training.generative;

import org.nd4j.linalg.api.ndarray.INDArray;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

public class GANVisualizationUtils {
    public static JFrame initFrame() {
        JFrame frame = new JFrame();
        frame.setTitle("Viz");
        frame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        frame.setLayout(new BorderLayout());
        return frame;
    }

    public static JPanel initPanel(JFrame frame, int numSamples) {
        JPanel panel = new JPanel();

        panel.setLayout(new GridLayout(numSamples / 4, 1, 8, 8));
        frame.add(panel, BorderLayout.CENTER);
        frame.setVisible(true);
        return panel;
    }

    public static void visualize(INDArray[] samples, JFrame frame, JPanel panel) {
        panel.removeAll();

        for (int i = 0; i < samples.length; i++) {
            panel.add(getImage(samples[i]));
        }

        frame.revalidate();
        frame.pack();
    }

    private static JLabel getImage(INDArray tensor) {
        BufferedImage bi = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        for (int i = 0; i < 784; i++) {
            int pixel = (int) (((tensor.getDouble(i) + 1) * 2) * 255);
            bi.getRaster().setSample(i % 28, i / 28, 0, pixel);
        }
        ImageIcon orig = new ImageIcon(bi);
        Image imageScaled = orig.getImage().getScaledInstance((8 * 28), (8 * 28), Image.SCALE_REPLICATE);

        ImageIcon scaled = new ImageIcon(imageScaled);

        return new JLabel(scaled);
    }
}
