/*
 * Copyright (c) 2020 CertifAI Sdn. Bhd.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package global.skymind.solution.image_processing.utils.display;

import org.nd4j.linalg.api.ndarray.INDArray;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

public class JPanelDisplay {
    private JFrame frame;
    private JPanel panel;
    private int gridWidth=1;
    private INDArray img;
    private int imgWidth;
    private int imgHeight;

    public JPanelDisplay(INDArray img, String caption)
    {
        this.img = img;
        this.imgHeight =  (int) img.size(2);
        this.imgWidth =  (int) img.size(3);
        frame = new JFrame();
        frame.setTitle(caption);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        panel = new JPanel();
        panel.setLayout(new GridLayout(0, gridWidth));

    }

    private JLabel getImage(){
        BufferedImage bi = new BufferedImage(this.imgWidth, this.imgHeight, BufferedImage.TYPE_BYTE_GRAY);
        for (int i = 0; i < imgWidth*imgHeight; i++) {
            int pixel = (int) img.getDouble(i);
            bi.getRaster().setSample(i % imgWidth, i / imgHeight, 0, pixel);
        }
        ImageIcon imgIcon = new ImageIcon(bi);
        return new JLabel(imgIcon);
    }

    public void display()
    {
        JLabel image = getImage();
        panel.add(image);
        frame.add(panel);
        frame.setVisible(true);
        frame.pack();
    }
}
