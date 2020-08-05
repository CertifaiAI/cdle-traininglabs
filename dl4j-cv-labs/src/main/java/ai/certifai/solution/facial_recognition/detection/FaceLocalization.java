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

package ai.certifai.solution.facial_recognition.detection;

public class FaceLocalization {
    float left_x;
    float left_y;
    float right_x;
    float right_y;

    public FaceLocalization(float left_x, float left_y, float right_x, float right_y){

        this.left_x = left_x;
        this.left_y = left_y;
        this.right_x = right_x;
        this.right_y = right_y;
    }

    public int getValidWidth(int imageWidth) {
        int width = (int) (this.right_x - this.left_x);
        if (( this.left_x + width) >= imageWidth){
            return (int) (imageWidth - this.left_x);
        }
        return width;
    }

    public int getValidHeight(int imageHeight){
        int height = (int) (this.right_y - this.left_y);
        if (( this.left_y + height)>= imageHeight){
            return (int) (imageHeight - this.left_y);
        }
        return height;
    }

    public int getWidth() {
        return (int) (this.right_x - this.left_x);
    }

    public int getHeight(){
        return (int) (this.right_y - this.left_y);
    }

    public float getLeft_x(){
        if (this.left_x < 0 ){
            this.left_x = 0;
        }
        return this.left_x;
    }

    public float getLeft_y(){
        if (this.left_y < 0){
            this.left_y = 0;
        }
        return this.left_y;
    }

    public float getRight_x(){
        return this.right_x;
    }

    public float getRight_y(){
        return this.right_y;
    }
}
