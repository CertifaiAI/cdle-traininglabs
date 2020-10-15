
Copyright (c) 2020 CertifAI Sdn. Bhd.
======================================
This program and the accompanying materials are made available under the
terms of the Apache License, Version 2.0 which is available at
https://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.

##### SPDX-License-Identifier: Apache-2.0


This folder contains everything you need to start a standalone DL4J project.

You can take this folder and use it as a base skeleton for any DL4J project. All fundamental DL4J dependencies are included in the pom.xml

You can run this on an IDE or using java terminal command.

## To run on IDE
1. Import project
2. Wait for IDE to resolve dependencies
3. Navigate to ```MyFirstDL4JProject.java``` 
4. Run program

## To run from command line
Firstly, the project needs to be compiled as a jar file. The command used will build an uber jar. This type of jar compiles all classes from this project with its dependencies.

### To build uber jar  
```
mvn clean package
```
The command will output .jar file in the ```target``` directory.

### Run program
```
cd target
java -cp my-first-dl4j-project-1.0-SNAPSHOT-bin.jar ai.certifai.MyFirstDL4JProject
```
MyFirstDL4JProject is the class to run which is located in ai.certifai package