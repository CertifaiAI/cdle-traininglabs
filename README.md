# Eclipse Deeplearning4j Training Labs

<p>
  <p align="center">
    <a href="https://sonarcloud.io/dashboard?id=CertifaiAI_TrainingLabs">
        <img alt="Sonar Cloud" src="https://sonarcloud.io/images/project_badges/sonarcloud-white.svg">
    </a>
</p>

<p>
  <p align="center">
    <a href="https://github.com/CertifaiAI/TrainingLabs/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/CertifaiAI/TrainingLabs.svg">
    </a>
    <a href="https://discord.com/invite/WsBFgNP">
        <img alt="Discord" src="https://img.shields.io/discord/699181979316387842?color=red">
    </a>
    <a href="https://certifai.ai">
        <img alt="Documentation" src="https://img.shields.io/website/https/certifai.ai.svg?color=ff69b4">
    </a>
    <a href="https://github.com/CertifaiAI/TrainingLabs/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/CertifaiAI/TrainingLabs.svg">
    </a>
    <a href="https://sonarcloud.io/dashboard?id=CertifaiAI_TrainingLabs">
        <img alt="Sonar Cloud" src="https://sonarcloud.io/api/project_badges/measure?project=CertifaiAI_TrainingLabs&metric=alert_status">
    </a>
</p>

Running examples strategically structured to enhance understanding of building models with Eclipse Deeplearning4j.

This repo contains codes for hands-on purpose during training session. All codes had been tested using CPU.

## Contents

#### DL4J-Lab
- [classification](https://github.com/CertifaiAI/TrainingLabs/tree/master/dl4j-labs/src/main/java/ai/certifai/solution/classification):
  This folder contains various binary and multiple classification task exercises for learners to practice and solutions to refer to.
- [convolution](https://github.com/CertifaiAI/TrainingLabs/tree/master/dl4j-labs/src/main/java/ai/certifai/solution/convolution):
  This folder contains MNIST classification and object detection using transfer learning of TinyYOLO and VGG16.
- [dataexamples](https://github.com/CertifaiAI/TrainingLabs/tree/master/dl4j-labs/src/main/java/ai/certifai/solution/dataexamples):
  This folder contains model that train neural network to learn drawing.
- [datavec](https://github.com/CertifaiAI/TrainingLabs/tree/master/dl4j-labs/src/main/java/ai/certifai/solution/datavec):
  This folder contains K-Fold cross-validation tutorial, load and transform CSV and Image files using DataVec.
- [feedforward](https://github.com/CertifaiAI/TrainingLabs/tree/master/dl4j-labs/src/main/java/ai/certifai/solution/feedforward):
  This folder contains simplest neural network to approximate a function that can map input to an output, and model that can detect the gender of a person based on his/her name.
- [generative](https://github.com/CertifaiAI/TrainingLabs/tree/master/dl4j-labs/src/main/java/ai/certifai/solution/generative):
  This folder contains a GAN model that can generate MNIST digits. 
- [humanactivity](https://github.com/CertifaiAI/TrainingLabs/tree/master/dl4j-labs/src/main/java/ai/certifai/solution/humanactivity):
  This folder contains a LSTM model and a hybrid CNN-LSTM neural network to perform human activity classification.
- [modelsaveload](https://github.com/CertifaiAI/TrainingLabs/tree/master/dl4j-labs/src/main/java/ai/certifai/solution/modelsaveload):
  This folder contains step-by-step on how to save and load a model.
- [nd4j](https://github.com/CertifaiAI/TrainingLabs/tree/master/dl4j-labs/src/main/java/ai/certifai/solution/nd4j):
  This folder contains ND4J tutorial and exercise.
- [recurrent](https://github.com/CertifaiAI/TrainingLabs/tree/master/dl4j-labs/src/main/java/ai/certifai/solution/recurrent):
  This folder contains a basic RNN network that learn to create a string, a LSTM model to generate texts, and a LSTM model for mortality prediction.
- [regression](https://github.com/CertifaiAI/TrainingLabs/tree/master/dl4j-labs/src/main/java/ai/certifai/solution/regression):
  This folder contains neural network that predict ridership demand.
- [VAE](https://github.com/CertifaiAI/TrainingLabs/tree/master/dl4j-labs/src/main/java/ai/certifai/solution/VAE):
  This folder contains a VAE model for bank transaction anomaly detection.
#### DL4J-cv-labs
- [classification](https://github.com/CertifaiAI/TrainingLabs/tree/master/dl4j-cv-labs/src/main/java/ai/certifai/solution/classification):
  This folder contains a custom model and a VGG16 model for dog breed classification, and a ResNet-50 model for food classification.
- [facial_recognition](https://github.com/CertifaiAI/TrainingLabs/tree/master/dl4j-cv-labs/src/main/java/ai/certifai/solution/facial_recognition):
  This folder contains facial recognition with a pipeline of video streaming, face detection and face recognition.
- [image processing](https://github.com/CertifaiAI/TrainingLabs/tree/master/dl4j-cv-labs/src/main/java/ai/certifai/solution/image_processing): 
  This folder contains step-by-step on image processing.
- [object_detection](https://github.com/CertifaiAI/TrainingLabs/tree/master/dl4j-cv-labs/src/main/java/ai/certifai/solution/object_detection): 
  This folder contains a TinyYOLO model and a YOLOv2 model for Avocado and Banana Object Detection, and a pre-trained YOLOv2 model for metal surface defects detection.
- [segmentation](https://github.com/CertifaiAI/TrainingLabs/tree/master/dl4j-cv-labs/src/main/java/ai/certifai/solution/segmentation): 
  This folder contains semantic segmentation on the car images and the cell nucleus images from Data Science Bowl 2018 using a pre-trained U-Net.


## Built with
- deeplearning4j 1.0.0-M1.1
- CUDA 11.2 (Note: Optional if you are using CPU)
- cuDNN 8.1.1 (Note: Optional if you are using CPU)

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
   -  CUDA 11.2
   -  cuDNN 8.1.1
  
CUDA and cuDNN can be downloaded from [here](https://developer.nvidia.com/cuda-11.2.0-download-archive) and [here](https://developer.nvidia.com/rdp/cudnn-archive). Step by step installation guides can be found [here](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html).

2. Dependencies are needed to be included into Maven project if we wish to use GPU for training. Follow the links below for instructions in details.
   - [ND4J backends for GPUs](https://deeplearning4j.konduit.ai/multi-project/explanation/configuration/backends#nd-4-j-backends-for-gpus-and-cpus)
   - [Using Deeplearning4J with cuDNN](https://deeplearning4j.konduit.ai/multi-project/explanation/configuration/backends/cudnn#using-cudnn-via-deeplearning-4-j)

## Usage
All examples are separated into [training](https://github.com/CertifaiAI/TrainingLabs/tree/master/dl4j-labs/src/main/java/ai/certifai/training) and [solution](https://github.com/CertifaiAI/TrainingLabs/tree/master/dl4j-labs/src/main/java/ai/certifai/solution) folders. The download will take some time to download dependencies from maven when you first run these examples.

All codes in <b>training</b> folder have few lines commented out so that they can be taught and demonstrated in the class. The <b>solution</b> folder contains the un-commented version for every line of codes.

For bad internet connection and unable to perform task smoothly, you can go to <b>src\main\resources\config.properties</b> and download necessary dataset before the sessions.

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

## Contributor's Guide
For contributors or someone who wishes to contribute, please take a look at the guideline [here](https://github.com/CertifaiAI/TrainingLabs/wiki/Contributor's-Guide) to help you in your journey.















