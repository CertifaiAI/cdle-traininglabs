/*
 * Copyright (c) 2020-2021 CertifAI Sdn. Bhd.
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
 *
 */

package ai.certifai.solution.object_detection.ActorsDetector.tinyyolo.dataHelpers;

import org.apache.commons.io.FilenameUtils;
import org.datavec.image.recordreader.objdetect.ImageObject;
import org.datavec.image.recordreader.objdetect.ImageObjectLabelProvider;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class LabelImgXmlLabelProvider implements ImageObjectLabelProvider {
    private Map<String, List<ImageObject>> labelMap = new HashMap();

    public LabelImgXmlLabelProvider(File dir) throws IOException, ParserConfigurationException, SAXException {
        File[] listOfFiles = dir.listFiles();

        for(int i = 0; i < listOfFiles.length; i++){
            String filename = listOfFiles[i].getName();
            if(filename.endsWith(".xml")||filename.endsWith(".XML"))
            {
                DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
                DocumentBuilder builder = factory.newDocumentBuilder();
                Document doc = builder.parse(listOfFiles[i]);
                NodeList objects = doc.getElementsByTagName("object");
                int m = objects.getLength();
                ArrayList<ImageObject> list = new ArrayList(m);
                for(int j = 0; j < m; ++j) {
                    Element el = (Element)objects.item(j);
                    String name = el.getElementsByTagName("name").item(0).getTextContent();
                    int xmin = Integer.parseInt(el.getElementsByTagName("xmin").item(0).getTextContent());
                    int ymin = Integer.parseInt(el.getElementsByTagName("ymin").item(0).getTextContent());
                    int xmax = Integer.parseInt(el.getElementsByTagName("xmax").item(0).getTextContent());
                    int ymax = Integer.parseInt(el.getElementsByTagName("ymax").item(0).getTextContent());
                    list.add(new ImageObject(xmin, ymin, xmax, ymax, name));
                }
                this.labelMap.put(FilenameUtils.getBaseName(filename), list);
            }
        }
    }

    public List<ImageObject> getImageObjectsForPath(String path) {
        File file = new File(path);
        String filename = file.getName();
        return (List)this.labelMap.get(FilenameUtils.getBaseName(filename));
    }

    public List<ImageObject> getImageObjectsForPath(URI uri) {
        return this.getImageObjectsForPath(uri.toString());
    }
}
