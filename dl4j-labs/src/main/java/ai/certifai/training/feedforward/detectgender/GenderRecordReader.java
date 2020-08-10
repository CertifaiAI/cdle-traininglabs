/*
 * Copyright (c) 2019 Skymind AI Bhd.
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

package ai.certifai.training.feedforward.detectgender;

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.reader.impl.LineRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.InputStreamInputSplit;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.nio.charset.Charset;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * GenderRecordReader class does following job
 * - Initialize method reads .CSV file as specified in Labels in constructor
 * - It loads person name and gender data into binary converted data
 *
 * - creates binary string iterator which can be used by RecordReaderDataSetIterator
 * - Generate list that contains actual binary data generated from person name, it also contains label (1 or 0) at the end
 *
 * - labels provided from the constructor
 * - then the pointed directory must have all of the label.csv
 */

public class GenderRecordReader extends LineRecordReader
{

    // list to hold labels passed in constructor
    private List<String> labels;

    // holds length of largest name out of all person names
    private int maxLengthName = 0;

    // String to hold all possible alphabets from all person names in raw data
    // This String is used to convert person name to binary string seperated by comma
    private String possibleCharacters = "";

    //iter and nameNLabelList is basically the same thing.
    // iterator for List "names" to be used in next() method
    private Iterator<String> iter;

    // Final list that contains actual binary data generated from person name, it also contains label (1 or 0) at the end
    private List<String> nameNLabelList = new ArrayList<>();

    //names
    List<String> names = new ArrayList<>();
    //binary names
    List<String> binaryNames = new ArrayList<>();


    // holds total number of names including both male and female names.
    // This variable is not used in PredictGenderTrain.java
    private int totalRecords = 0;

    /**
     * Constructor to allow client application to pass List of possible Labels
     * @param labels - List of String that client application pass all possible labels, in our case "M" and "F"
     */
    public GenderRecordReader(List<String> labels)
    {
        this.labels = labels;
    }

    /**
     * This function does following steps
     * - Looks for the files with the name (in specified folder) as specified in labels set in constructor
     * - File must have person name and gender of the person (M or F),
     *   e.g. Deepan,M
     *        Trupesh,M
     *        Vinay,M
     *        Ghanshyam,M
     *
     *        Meera,F
     *        Jignasha,F
     *        Chaku,F
     *
     * - File for male and female names must be different, like M.csv, F.csv etc.
     * - populates all names in temporary list
     * - generate binary string for each alphabet for all person names
     * - combine binary string for all alphabets for each name
     * - find all unique alphabets to generate binary string mentioned in above step
     * - take equal number of records from all files. To do that, finds minimum record from all files, and then takes
     *   that number of records from all files to keep balance between data of different labels.
     * - Note : this function uses stream() feature of Java 8, which makes processing faster. Standard method to process file takes more than 5-7 minutes.
     *          using stream() takes approximately 800-900 ms only.
     * - Final converted binary data is stored List<String> names variable
     * - sets iterator from "names" list to be used in next() function
     * @param split - user can pass directory containing .CSV file for that contains names of male or female
     * @throws IOException
     * @throws InterruptedException
     */
    @Override
    public void initialize(InputSplit split) throws IOException, InterruptedException {
        if (split instanceof FileSplit) {
            URI[] locations = split.locations();

            if (locations != null && locations.length > 1) {
                System.out.println("Input files: ");

                String longestName = "";
                String uniqueCharactersTempString = "";

                //<list of different labels <label name, list of names>>
                List<Pair<String, List<String>>> labelNameList = new ArrayList<>();

                for (URI location : locations) {
                    File file = new File(location);

                    System.out.println("File: " + file.toString());

                    List<String> temp = this.labels.stream().filter(line -> file.getName().equals(line + ".csv")).collect(Collectors.toList());

                    if (temp.size() > 0) {

                        Path path = Paths.get(file.getAbsolutePath());

                        //List<String> names = java.nio.file.Files.readAllLines(path, Charset.defaultCharset()).stream().map(element -> element.split(",")[0]).collect(Collectors.toList());
                        List<String> tempName = java.nio.file.Files.readAllLines(path, Charset.defaultCharset()).stream().map(element -> element.split("\n")[0]).collect(Collectors.toList());

                        List<String> names = new ArrayList<>();

                        for(String i : tempName)
                        {
                            names.add(i.toLowerCase());
                        }

                        Optional<String> optional = names.stream().reduce((name1, name2) -> name1.length() >= name2.length() ? name1 : name2);
                        if (optional.isPresent() && optional.get().length() > longestName.length()) {
                            longestName = optional.get();
                        }

                        uniqueCharactersTempString = uniqueCharactersTempString + names.toString();

                        //Label, list of names
                        Pair<String, List<String>> nameList = new Pair<>(temp.get(0), names);
                        labelNameList.add(nameList);
                    } else {
                        throw new InterruptedException("File missing for any of the specified labels");
                    }
                }


                this.maxLengthName = longestName.length();

                //get unique characters from example [alex, karev][christine, jolie], trim away "[" and "]"
                String unique = Stream.of(uniqueCharactersTempString).map(w -> w.split("")).flatMap(Arrays::stream).distinct().collect(Collectors.toList()).toString();

                char[] chars = unique.toCharArray();
                Arrays.sort(chars);

                unique = new String(chars);
                unique = unique.replaceAll("\\[", "").replaceAll("\\]","").replaceAll(",","");
                if(unique.startsWith(" "))
                    unique = " " + unique.trim();

                this.possibleCharacters = unique;


                //compare to get minimum number of data between diff labels
                int minSize = Integer.MAX_VALUE;
                for(Pair<String, List<String>> tempPair : labelNameList)
                {
                    int currentSize = tempPair.getSecond().size();
                    if (minSize > currentSize)
                    {
                        minSize = currentSize;
                    }

                }

                //********************To equal out the data of each label but cutting out the extra
                List<Pair<String, List<String>>> temp1 = new ArrayList<>();
                for(Pair<String, List<String>> i : labelNameList)
                {
                    int diff = Math.abs(minSize - i.getSecond().size());
                    List<String> tempList = new ArrayList<String>();

                    if (i.getSecond().size() > minSize)
                    {
                        tempList = i.getSecond();
                        tempList = tempList.subList(0, tempList.size() - diff);
                    }
                    else
                    {
                        tempList = i.getSecond();
                    }

                    Pair<String, List<String>> tempNewPair = new Pair<>(i.getFirst(), tempList);
                    temp1.add(tempNewPair);
                    names.addAll(tempList);
                }

                labelNameList.clear();

                //********************transform string to binary and add 0 or 1 behind as label

                List<Pair<String, List<String>>> temp2 = new ArrayList<>();

                for(int i = 0; i < temp1.size() ; i++)
                {
                    int gender = temp1.get(i).getFirst().equals("M") ? 1 : 0;
                    List<String> secondList = temp1.get(i).getSecond().stream().map(element -> getBinaryString(element, gender)).collect(Collectors.toList());
                    Pair<String,List<String>> secondTempPair = new Pair<>(temp1.get(i).getFirst(),secondList);
                    temp2.add(secondTempPair);
                    binaryNames.addAll(secondList);
                }
                temp1.clear();

                //********************
                for(Pair<String, List<String>> i : temp2)
                {
                    nameNLabelList.addAll(i.getSecond());
                }
                temp2.clear();

                //********************
                this.totalRecords = nameNLabelList.size();
                Collections.shuffle(nameNLabelList);
                this.iter = nameNLabelList.iterator();

            }

            /*
            System.out.println("Longest length name: " +this.maxLengthName);
            System.out.println("Total unique names: " + this.totalRecords);

            for(String i : nameNLabelList)
            {
                System.out.println(i);
            }
            */




        } else if (split instanceof InputStreamInputSplit) {
            System.out.println("InputStream Split found...Currently not supported");
            throw new InterruptedException("File missing for any of the specified labels");
        }


    }


    /**
     * This function gives binary string for full name string
     * - It uses "PossibleCharacters" string to find the decimal equivalent to any alphabet from it
     * - generate binary string for each alphabet
     * - left pads binary string for each alphabet to make it of size 5
     * - combine binary string for all alphabets of a name
     * - Right pads complete binary string to make it of size that is the size of largest name to keep all name length of equal size
     * - appends label value (1 or 0 for male or female respectively)
     * @param name - person name to be converted to binary string
     * @param gender - variable to decide value of label to be added to name's binary string at the end of the string
     * @return
     */
    private String getBinaryString(String name, int gender)
    {
        String binaryString = "";
        for (int j = 0; j < name.length(); j++)
        {
            String fs = org.apache.commons.lang3.StringUtils.leftPad(Integer.toBinaryString(this.possibleCharacters.indexOf(name.charAt(j))),5,"0");
            binaryString = binaryString + fs;
        }
        //binaryString = String.format("%-" + this.maxLengthName*5 + "s",binaryString).replace(' ','0'); // this takes more time than StringUtils, hence commented

        binaryString  = org.apache.commons.lang3.StringUtils.rightPad(binaryString,this.maxLengthName*5,"0");
        binaryString = binaryString.replaceAll(".(?!$)", "$0,");

        //System.out.println("binary String : " + binaryString);
        return binaryString + "," + String.valueOf(gender);
    }

    public List<String> getNames()
    {
        return this.names;
    }

    public List<String> getBinaryNames()
    {
        return this.binaryNames;
    }

    public int getNameMaxLength()
    {
        return maxLengthName;
    }

    /**
     * - takes onme record at a time from names list using iter iterator
     * - stores it into Writable List and returns it
     *
     * @return
     */
    @Override
    public List<Writable> next()
    {
        if (iter.hasNext())
        {
            List<Writable> ret = new ArrayList<>();
            String currentRecord = iter.next();
            String[] temp = currentRecord.split(",");

            for(int i = 0;i < temp.length; i++)
            {
                ret.add(new DoubleWritable(Double.parseDouble(temp[i])));
            }
            return ret;
        }
        else
            throw new IllegalStateException("no more elements");
    }

    @Override
    public boolean hasNext()
    {
        if(iter != null) {
            return iter.hasNext();
        }
        throw new IllegalStateException("Indeterminant state: record must not be null, or a file iterator must exist");
    }

    @Override
    public void close() throws IOException {

    }

    @Override
    public void setConf(Configuration conf) {
        this.conf = conf;
    }

    @Override
    public Configuration getConf() {
        return conf;
    }

    @Override
    public void reset()
    {
        this.iter = nameNLabelList.iterator();
    }


}
