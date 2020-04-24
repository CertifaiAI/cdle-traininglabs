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

package global.skymind;

import org.bytedeco.javacv.FrameFilter;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Properties;

public class Helper {

    public static String getPropValues(String property) throws IOException {
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

    public static String getCheckSum(String filePath) {
        String hashValue = "";
        try (InputStream is = Files.newInputStream(Paths.get(filePath))) {
            hashValue = org.apache.commons.codec.digest.DigestUtils.md5Hex(is);
        } catch (Exception e){
            System.out.println(e.getMessage());
        }
        return hashValue;
    }
}
