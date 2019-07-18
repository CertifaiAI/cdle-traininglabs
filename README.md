# DL4J Training Labs Materials
This repo contains codes for hands-on purpose during training session.
## Contents

#### DL4J-Lab
- [convolution/MNIST](https://github.com/skymindglobal/TrainingLabs/tree/master/dl4j-labs/src/main/java/global/skymind/solution/convolution/mnist):
  Mnist classification using CNN.
- [convolution/TransferLearning/TinyYoLo/TLDetectorActors](https://github.com/skymindglobal/TrainingLabs/tree/master/dl4j-labs/src/main/java/global/skymind/solution/convolution/objectdetection/transferlearning/tinyyolo):Detect
  actors face using CNN.
- [convolution/VGG16](https://github.com/skymindglobal/TrainingLabs/tree/master/dl4j-labs/src/main/java/global/skymind/solution/convolution/objectdetection/transferlearning/vgg16):
  Image classification on oil palm images.
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

#### Spark-Lab
- [HelloWorldSpark](https://github.com/skymindglobal/TrainingLabs/tree/master/dl4j-spark-labs/src/main/java/global/skymind)

## Built with
deeplearning4j beta 4.0

## Getting Started ##

### Install Java ###

Download Java JDK
[here](https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html)\.
Note that DL4J supports Java 1.7 or later.

Check the version of Java using: 
```sh
java -version
```

Make sure that 64- Bit version of Java is installed.

### Install IntelliJ IDEA Community Edition ###
Download and install 
[IntelliJ IDEA](https://www.jetbrains.com/idea/download/).

### Install Apache Maven * ###

<b>\* Optional if you are using prebuilt maven from IntelliJ </b>

Follow these [instructions](https://maven.apache.org/install.html) to install Apache Maven.

## Usage ##
All examples are separated into
[trainings](https://github.com/skymindglobal/TrainingLabs/tree/master/dl4j-labs/src/main/java/global/skymind/training)
and
[solutions](https://github.com/skymindglobal/TrainingLabs/tree/master/dl4j-labs/src/main/java/global/skymind/solution)
folders. The download will take sometime to download dependencies from
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
















   