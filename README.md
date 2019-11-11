# DL4J Training Labs Materials
This repo contains codes for hands-on purpose during training session.
## Contents

#### DL4J-Lab
- [convolution/MNIST](https://github.com/skymindglobal/TrainingLabs/tree/master/dl4j-labs/src/main/java/global/skymind/solution/convolution/mnist):
  Mnist classification using CNN.
- [convolution/TransferLearning/TinyYoLo/TLDetectorActors](https://github.com/skymindglobal/TrainingLabs/tree/master/dl4j-labs/src/main/java/global/skymind/solution/convolution/objectdetection/transferlearning/tinyyolo):Detect
  actors face using CNN.
- [dataexample/ImageDrawer](https://github.com/skymindglobal/TrainingLabs/tree/master/dl4j-labs/src/main/java/global/skymind/solution/dataexamples):
  Train neural network that learns to draw.
- [feedforward/detectgender](https://github.com/skymindglobal/TrainingLabs/tree/master/dl4j-labs/src/main/java/global/skymind/solution/feedforward/detectgender):
  Detect the gender of a person based on his/her name.
- [feedforward/SimplestNetwork](https://github.com/skymindglobal/TrainingLabs/tree/master/dl4j-labs/src/main/java/global/skymind/solution/feedforward):
  Simplest neural network to approximate a function that can map input
  to an output.
- [generative/MnistGAN](https://github.com/skymindglobal/TrainingLabs/tree/master/dl4j-labs/src/main/java/global/skymind/solution/generative):
  A GAN model to generate mnist digits. 
- [humanactivity/CNNLSTM](https://github.com/skymindglobal/TrainingLabs/tree/master/dl4j-labs/src/main/java/global/skymind/solution/humanactivity
                                                                                                                                                    ):
  A hybrid CNN-LSTM neural network to perform human activity
  classification.
- [humanactivity/LSTM](https://github.com/skymindglobal/TrainingLabs/tree/master/dl4j-labs/src/main/java/global/skymind/solution/humanactivity):
  An LSTM model to classify human activity.
- [modelsaveload/MnistImageLoad](https://github.com/skymindglobal/TrainingLabs/tree/master/dl4j-labs/src/main/java/global/skymind/solution/modelsaveload):
  Step-by-step on how to save a model.
- [modelsaveload/MnistImageSave](https://github.com/skymindglobal/TrainingLabs/tree/master/dl4j-labs/src/main/java/global/skymind/solution/modelsaveload):
  Step-by-step on how to load a model.
- [recurrent/basic](https://github.com/skymindglobal/TrainingLabs/tree/master/dl4j-labs/src/main/java/global/skymind/solution/recurrent/basic):
  Basic RNN network that learns to create a string.
- [recurrent/character](https://github.com/skymindglobal/TrainingLabs/tree/master/dl4j-labs/src/main/java/global/skymind/solution/recurrent/character):
  Texts generation using LSTM.
- [recurrent/physionet](https://github.com/skymindglobal/TrainingLabs/tree/master/dl4j-labs/src/main/java/global/skymind/solution/recurrent/physionet):
  Mortality prediction using LSTM.
- [VAE/VAECreditAnomaly](https://github.com/skymindglobal/TrainingLabs/tree/master/dl4j-labs/src/main/java/global/skymind/solution/VAE):
  Bank transaction anomaly detection using VAE.
- [ND4J](https://github.com/skymindglobal/TrainingLabs/tree/master/dl4j-labs/src/main/java/global/skymind/solution/nd4j):
  ND4J tutorial and exercise.
- [DataVec](https://github.com/skymindglobal/TrainingLabs/tree/datavec/dl4j-labs/src/main/java/global/skymind/solution/datavec):
  Load and transform CSV and Image files using DataVec
  
#### DL4J-cv-labs
- [ImageClassification/CustomModel](https://github.com/skymindglobal/TrainingLabs/tree/imageclassification/dl4j-cv-labs/src/main/java/global/skymind/solution/classification):
  Dog breed classification using custom model.
- [ImageClassification/TransferLearning](https://github.com/skymindglobal/TrainingLabs/tree/imageclassification/dl4j-cv-labs/src/main/java/global/skymind/solution/classification/transferlearning):
  Dog breed classification using transfer learning.
- [ObjectDetection/YOLO](https://github.com/skymindglobal/TrainingLabs/tree/master/dl4j-cv-labs/src/main/java/global/skymind/solution/object_detection): Avocado and Banana Object Detection model using Transfer learning of TinyYOLO and YOLOv2.
- [segmentation/PretrainedUNET](https://github.com/skymindglobal/TrainingLabs/tree/segmentation/dl4j-cv-labs/src/main/java/global/skymind/solution/segmentation/PretrainedUNET): Semantic segmentation on the Cell nucleus image from Data Science Bowl 2018, using a Pre-trained U-Net.
- [segmentation/ImageAugmentation](https://github.com/skymindglobal/TrainingLabs/tree/segmentation/dl4j-cv-labs/src/main/java/global/skymind/solution/segmentation/ImageAugmentation): (Optional) Image augmentation to increase samples, if required.
#### Spark-Lab
- [HelloWorldSpark](https://github.com/skymindglobal/TrainingLabs/tree/master/dl4j-spark-labs/src/main/java/global/skymind)

## Built with
- deeplearning4j beta 4.0
- CUDA 10.0 (Note: Optional if you are using CPU)
- cuDNN 7.6 (Note: Optional if you are using CPU)

## Getting Started ##

### Install Java ###

Download Java JDK
[here](https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html).  
(Note: Use Java 8 for full support of DL4J operations)

Check the version of Java using: 
```sh
java -version
```

Make sure that 64-Bit version of Java is installed.

### Install IntelliJ IDEA Community Edition ###
Download and install 
[IntelliJ IDEA](https://www.jetbrains.com/idea/download/).

### Install Apache Maven  *Optional* ###
IntelliJ provides a default Maven that is bundled with the installer.
Follow these [instructions](https://maven.apache.org/install.html) to install Apache Maven.

### GPU setup  *Optional* ##
Follow the instructions below if you plan to use GPU setup.
1. Install CUDA and cuDNN <br> 
    Requirements:
   -  CUDA 10.0 
   -  cuDNN 7.6


CUDA and cuDNN can be downloaded from
[here](https://developer.nvidia.com/cuda-10.0-download-archive) and
[here](https://developer.nvidia.com/cudnn). Step by step installation
guides can be found
[here](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html).

2. Dependencies are needed to be included into Maven project if we wish
   to use GPU for training. Follow the links below for instructions in
   details.
   -  [ND4J backends for GPUs](https://deeplearning4j.org/docs/latest/deeplearning4j-config-gpu-cpu)
   - [Using Deeplearning4J with cuDNN](https://deeplearning4j.org/docs/latest/deeplearning4j-config-cudnn)
## Usage ##
All examples are separated into
[training](https://github.com/skymindglobal/TrainingLabs/tree/master/dl4j-labs/src/main/java/global/skymind/training)
and
[solution](https://github.com/skymindglobal/TrainingLabs/tree/master/dl4j-labs/src/main/java/global/skymind/solution)
folders. The download will take some time to download dependencies from
maven when you first run these examples.

All codes in <b>training</b> folder have few lines commented out so that
they can be taught and demonstrated in the class. The <b>solution</b>
folder contains the un-commented version for every line of codes.


## Known Issues ##
<b>Problem</b>: 
```sh
jnind4jcpu.dll: Can't find dependent libraries
```
<b>Solution</b>: <br /> Change the maven dependencies of Javacpp to the
latest (1.4.3 works).

<b>Problem</b>: 
```sh
"C:\Users\LohJZ\.javacpp\cache\cuda-10.0-7.3-1.4.3-windows-x86_64.jar\org\bytedeco\javacpp\windows-x86_64\jnicudnn.dll": Can't find procedure
```
<b>Solution</b>: <br /> Install latest CUDA (version 7.6 works)
















   
