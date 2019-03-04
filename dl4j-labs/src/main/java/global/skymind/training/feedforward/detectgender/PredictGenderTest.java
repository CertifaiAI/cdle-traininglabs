package global.skymind.training.feedforward.detectgender;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

//maxLengthName is the longest name trained. need to manually altered
public class PredictGenderTest implements Runnable {
    private int row=0;
    private JDialog jd;
    private JTextField jtf;
    private JLabel jlbl;
    private String possibleCharacters;
    private JLabel gender;
    private String filePath;
    private JButton btnNext;
    private JLabel genderLabel;
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
            this.filePath = new ClassPathResource("PredictGender").getFile().getAbsolutePath() + "/Data/";
            this.possibleCharacters = " abcdefghijklmnopqrstuvwxyz";
            this.model = ModelSerializer.restoreMultiLayerNetwork(System.getProperty("java.io.tmpdir") + "PredictGender.zip");
        }
        catch(Exception e)
        {
            System.out.println("Exception : " + e.getMessage());
        }
    }
}
