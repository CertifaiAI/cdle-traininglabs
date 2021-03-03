# Eclipse Deeplearning4j Training Labs

<p>
  <p align="center">
    <a href="https://github.com/CertifaiAI/TrainingLabs/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/CertifaiAI/TrainingLabs.svg">
    </a>
    <a href="Discord">
        <img alt="Discord" src="https://img.shields.io/discord/699181979316387842?color=red">
    </a>
    <a href="https://certifai.ai">
        <img alt="Documentation" src="https://img.shields.io/website/http/certifai.ai.svg?color=ff69b4">
    </a>
    <a href="https://github.com/CertifaiAI/TrainingLabs/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/CertifaiAI/TrainingLabs.svg">
    </a>
</p>

Running examples strategically structured to enhance understanding of building models with Eclipse Deeplearning4j.

[![](https://sourcerer.io/fame/chiaweilim/skymindglobal/TrainingLabs/images/0)](https://sourcerer.io/fame/chiaweilim/skymindglobal/TrainingLabs/links/0)[![](https://sourcerer.io/fame/chiaweilim/skymindglobal/TrainingLabs/images/1)](https://sourcerer.io/fame/chiaweilim/skymindglobal/TrainingLabs/links/1)[![](https://sourcerer.io/fame/chiaweilim/skymindglobal/TrainingLabs/images/2)](https://sourcerer.io/fame/chiaweilim/skymindglobal/TrainingLabs/links/2)[![](https://sourcerer.io/fame/chiaweilim/skymindglobal/TrainingLabs/images/3)](https://sourcerer.io/fame/chiaweilim/skymindglobal/TrainingLabs/links/3)[![](https://sourcerer.io/fame/chiaweilim/skymindglobal/TrainingLabs/images/4)](https://sourcerer.io/fame/chiaweilim/skymindglobal/TrainingLabs/links/4)[![](https://sourcerer.io/fame/chiaweilim/skymindglobal/TrainingLabs/images/5)](https://sourcerer.io/fame/chiaweilim/skymindglobal/TrainingLabs/links/5)[![](https://sourcerer.io/fame/chiaweilim/skymindglobal/TrainingLabs/images/6)](https://sourcerer.io/fame/chiaweilim/skymindglobal/TrainingLabs/links/6)[![](https://sourcerer.io/fame/chiaweilim/skymindglobal/TrainingLabs/images/7)](https://sourcerer.io/fame/chiaweilim/skymindglobal/TrainingLabs/links/7)

This repo contains codes for hands-on purpose during training session. All codes had been tested using CPU.

## Contents
| Lab  | Description |
| ------------- | ------------- |
| dl4j-labs  | Tutorials covering the basics of DL4J and building Deep Learning models to solve simple classification and regression problems.  |
| dl4j-cv-labs  | Tutorials covering image processing topics and solving simple Computer Vision problems with Deep Learning models using DL4J.  |



## Built with
- deeplearning4j beta 7.0
- CUDA 10.1 (Note: Optional if you are using CPU)
- cuDNN 7.6 (Note: Optional if you are using CPU)

## Getting Started

### Install Java

Download Java JDK [here](https://www.oracle.com/java/technologies/javase/javase-jdk8-downloads.html).  
(Note: Use Java 8 for full support of DL4J operations)

Check the version of Java using: 
```sh
java -version
```

Make sure that 64-Bit version of Java is installed.

### Install IntelliJ IDEA Community Edition
Download and install [IntelliJ IDEA](https://www.jetbrains.com/idea/download/).

### Install Apache Maven  *Optional*
IntelliJ provides a default Maven that is bundled with the installer. Follow these [instructions](https://maven.apache.org/install.html) to install Apache Maven.

### GPU setup  *Optional*
Follow the instructions below if you plan to use GPU setup.
1. Install CUDA and cuDNN
    Requirements:
   -  CUDA 10.1
   -  cuDNN 7.6
  
CUDA and cuDNN can be downloaded from [here](https://developer.nvidia.com/cuda-10.1-download-archive-base) and [here](https://developer.nvidia.com/rdp/cudnn-archive). Step by step installation guides can be found [here](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html).

2. Dependencies are needed to be included into Maven project if we wish to use GPU for training. Follow the links below for instructions in details.
   - [ND4J backends for GPUs](https://deeplearning4j.konduit.ai/config/backends#nd-4-j-backends-for-gpus-and-cpus)
   - [Using Deeplearning4J with cuDNN](https://deeplearning4j.konduit.ai/config/backends/config-cudnn#using-deeplearning-4-j-with-cudnn)

## Usage
All examples are separated into [training](https://github.com/CertifaiAI/TrainingLabs/tree/master/dl4j-labs/src/main/java/ai/certifai/training) and [solution](https://github.com/CertifaiAI/TrainingLabs/tree/master/dl4j-labs/src/main/java/ai/certifai/solution) folders. The download will take some time to download dependencies from maven when you first run these examples.

All codes in <b>training</b> folder have few lines commented out so that they can be taught and demonstrated in the class. The <b>solution</b> folder contains the un-commented version for every line of codes.


## Known Issues
<b>Problem</b>: 
```sh
jnind4jcpu.dll: Can't find dependent libraries
```
<b>Solution</b>: <br /> Change the maven dependencies of Javacpp to the latest (1.4.3 works).

<b>Problem</b>: 
```sh
"C:\Users\LohJZ\.javacpp\cache\cuda-10.0-7.3-1.4.3-windows-x86_64.jar\org\bytedeco\javacpp\windows-x86_64\jnicudnn.dll": Can't find procedure
```
<b>Solution</b>: <br /> Install latest CUDA (version 7.6 works)

















