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

package ai.certifai.solution.feedforward.detectgender;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.nio.file.Paths;

//maxLengthName is the longest name trained. need to manually altered
public class PredictGenderTest implements Runnable {
    private JDialog jd;
    private JTextField jtf;
    private JLabel jlbl;
    private String possibleCharacters;
    private JLabel gender;
    private String filePath;
    private JButton btnNext;
    private MultiLayerNetwork model;
    private final int maxLengthName = 11;

    public static void main(String[] args) throws Exception
    {
        PredictGenderTest pgt = new PredictGenderTest();
        Thread t = new Thread(pgt);
        t.start();
        pgt.prepareInterface();
    }

    public void prepareInterface()
    {
        this.jd = new JDialog();
        this.jd.getContentPane().setLayout(null);
        this.jd.setBounds(100,100,300,200);
        this.jd.setLocationRelativeTo(null);
        this.jd.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        this.jd.setTitle("Predict Gender By Name");
        this.jd.setVisible(true);

        //jd.add(jp);

        this.jlbl = new JLabel();
        this.jlbl.setBounds(25,20,100,20);
        this.jlbl.setText("Enter Name : ");
        this.jd.add(jlbl);

        this.jtf = new JTextField();
        this.jtf.setBounds(105,20,150,20);
        this.jd.add(jtf);

        this.gender = new JLabel();
        this.gender.setBounds(105,22,75,120);
        this.jd.add(gender);

        this.btnNext = new JButton();
        this.btnNext.setBounds(25,120,150,20);
        this.btnNext.setText("Predict");


        this.btnNext.addActionListener(new ActionListener() {

            public void actionPerformed(ActionEvent e)
            {
                if (!jtf.getText().isEmpty()) {
                    String name = jtf.getText().toLowerCase();
                    System.out.println("Name: Entered: " + name);
                    String binaryData = getBinaryString(name);
                    //System.out.println("binaryData : " + binaryData);
                    String[] arr = binaryData.split(",");
                    INDArray features = Nd4j.zeros(1, maxLengthName * 5);

                    for (int i = 0; i < arr.length; i++)
                    {
                        features.putScalar(new int[]{0, i}, Integer.parseInt(arr[i]));
                    }

                    INDArray predicted = model.output(features);
                    //System.out.println("output : " + predicted);
                    if (predicted.getDouble(0) > predicted.getDouble(1))
                        gender.setText("Female");
                    else if (predicted.getDouble(0) < predicted.getDouble(1))
                        gender.setText("Male");
                    else
                        gender.setText("Both male and female can have this name");
                }
                else
                    gender.setText("Enter name please..");
            }
        });

        this.jd.add(this.btnNext);
    }

    private String getBinaryString(String name)
    {
        String binaryString = "";
        for (int j = 0; j < name.length(); j++)
        {
            String fs = org.apache.commons.lang3.StringUtils.leftPad(Integer.toBinaryString(this.possibleCharacters.indexOf(name.charAt(j))),5,"0");
            binaryString = binaryString + fs;
        }
        //binaryString = String.format("%-" + this.maxLengthName*5 + "s",binaryString).replace(' ','0'); // this takes more time than StringUtils, hence commented

        binaryString  = org.apache.commons.lang3.StringUtils.rightPad(binaryString, this.maxLengthName * 5,"0");
        binaryString = binaryString.replaceAll(".(?!$)", "$0,");

        //System.out.println("binary String : " + binaryString);
        return binaryString;
    }

    private String pad(String string,int total_length)
    {
        String str = string;
        int diff = 0;
        if(total_length > string.length())
            diff = total_length - string.length();
        for(int i=0;i<diff;i++)
            str = "0" + str;
        return str;
    }

    public void run()
    {
        try
        {
            this.filePath = Paths.get(System.getProperty("java.io.tmpdir") , "PredictGender.zip").toString();
            this.possibleCharacters = " abcdefghijklmnopqrstuvwxyz";
            this.model = ModelSerializer.restoreMultiLayerNetwork(this.filePath);
        }
        catch(Exception e)
        {
            System.out.println("Exception : " + e.getMessage());
        }
    }
}
