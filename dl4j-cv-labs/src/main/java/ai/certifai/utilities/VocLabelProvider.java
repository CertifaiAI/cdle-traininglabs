package ai.certifai.utilities;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;

import lombok.NonNull;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.datavec.image.recordreader.objdetect.ImageObject;
import org.datavec.image.recordreader.objdetect.ImageObjectLabelProvider;

public class VocLabelProvider implements ImageObjectLabelProvider {
    private String annotationsDir;

    public VocLabelProvider(@NonNull String baseDirectory) {
        this.annotationsDir = FilenameUtils.concat(baseDirectory, "Annotations");
        if (!(new File(this.annotationsDir)).exists())
        {
            throw new IllegalStateException("Annotations directory does not exist. Annotation files should be present at baseDirectory/Annotations/nnnnnn.xml. Expected location: " + this.annotationsDir);
        }
    }

    public List<ImageObject> getImageObjectsForPath(String path) {
        int idx = path.lastIndexOf('/');
        idx = Math.max(idx, path.lastIndexOf('\\'));
        String filename = path.substring(idx + 1, path.lastIndexOf('.'));
        String xmlPath = FilenameUtils.concat(this.annotationsDir, filename + ".xml");
        File xmlFile = new File(xmlPath);
        if (!xmlFile.exists()) {
            throw new IllegalStateException("Could not find XML file for image " + path + "; expected at " + xmlPath);
        } else {
            String xmlContent;
            try {
                xmlContent = FileUtils.readFileToString(xmlFile, "UTF-8");
            } catch (IOException var17) {
                throw new RuntimeException(var17);
            }

            String[] lines = xmlContent.split("\n");
            List<ImageObject> out = new ArrayList();

            for(int i = 0; i < lines.length; ++i) {
                if (lines[i].contains("<object>")) {
                    String name = null;
                    int xmin = -2147483648;
                    int ymin = -2147483648;
                    int xmax = -2147483648;
                    int ymax = -2147483648;

                    while(true) {
                        while(!lines[i].contains("</object>")) {
                            if (name == null && lines[i].contains("<name>")) {
                                int idxStartName = lines[i].indexOf(62) + 1;
                                int idxEndName = lines[i].lastIndexOf(60);
                                name = lines[i].substring(idxStartName, idxEndName);
                                ++i;
                            } else if (xmin == -2147483648 && lines[i].contains("<xmin>")) {
                                xmin = this.extractAndParse(lines[i]);
                                ++i;
                            } else if (ymin == -2147483648 && lines[i].contains("<ymin>")) {
                                ymin = this.extractAndParse(lines[i]);
                                ++i;
                            } else if (xmax == -2147483648 && lines[i].contains("<xmax>")) {
                                xmax = this.extractAndParse(lines[i]);
                                ++i;
                            } else if (ymax == -2147483648 && lines[i].contains("<ymax>")) {
                                ymax = this.extractAndParse(lines[i]);
                                ++i;
                            } else {
                                ++i;
                            }
                        }

                        if (name == null) {
                            throw new IllegalStateException("Invalid object format: no name tag found for object in file " + xmlPath);
                        }

                        if (xmin == -2147483648 || ymin == -2147483648 || xmax == -2147483648 || ymax == -2147483648) {
                            throw new IllegalStateException("Invalid object format: did not find all of xmin/ymin/xmax/ymax tags in " + xmlPath);
                        }

                        out.add(new ImageObject(xmin, ymin, xmax, ymax, name));
                        break;
                    }
                }
            }

            return out;
        }
    }

    private int extractAndParse(String line) {
        int idxStartName = line.indexOf('>') + 1;
        int idxEndName = line.lastIndexOf('<');
        String substring = line.substring(idxStartName, idxEndName);
        return Integer.parseInt(substring);
    }

    public List<ImageObject> getImageObjectsForPath(URI uri) {
        return this.getImageObjectsForPath(uri.toString());
    }
}
