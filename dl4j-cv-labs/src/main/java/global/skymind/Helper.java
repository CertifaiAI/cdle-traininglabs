package global.skymind;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

public class Helper {

    public static String getPropValues(String property)throws IOException {
        InputStream inputStream = null;
        String value = null;

        try{
            Properties prop = new Properties();
            String propFileName = "config.properties";

            inputStream = Helper.class.getClassLoader().getResourceAsStream(propFileName);

            if (inputStream != null) {
                prop.load(inputStream);
            } else {
                throw new FileNotFoundException("property file '" + propFileName + "' not found in the classpath");
            }

            value = prop.getProperty(property);
        } catch (Exception  e) {
            e.printStackTrace();
        } finally {
            inputStream.close();
        }

        return value;
    }
}
