<!-- # please respect the structure below-->
*See the [MediaEval 2022 webpage](https://multimediaeval.github.io/editions/2022/) for information on how to register and participate.* <br>
*See the [Sport Task MediaEval 2022 webpage](https://multimediaeval.github.io/editions/2022/tasks/sportsvideo/) for information on the task.*

# Introduction

This is our group's approach to the [Sport Task MediaEval 2022 benchmark competition](https://multimediaeval.github.io/editions/2022/tasks/sportsvideo/).

This task is divided into two Substasks:

***Subtask 1 :*** Participants are required to build a classification system that automatically labels video segments according to a performed stroke. There are 20 possible stroke classes and an additional non-stroke class.

***Subtask 2 :*** The goal here is to detect if a stroke has been performed, whatever its classes, and to extract its temporal boundaries. The aim is to be able to distinguish between moments of interest in a game (players performing strokes) from irrelevant moments (picking up the ball, having a break…). This subtask can be a preliminary step for later recognizing a stroke that has been performed.

# Baseline
In order to help participants in their submission, to process videos, annotation files and deep learning techniques, there is a given baseline. The method is simple and is based on a single stream 3D CNN with an attention mechanism using only RGB data.

Pierre-Etienne Martin, Jenny Benois-Pineau, Renaud Péteri, Julien Morlier. 3D attention mechanism for fine-grained classification of table tennis strokes using a Twin Spatio-Temporal Convolutional Neural Networks. 25th International Conference on Pattern Recognition (ICPR2020), Jan 2021, Milano, Italy. [⟨hal-02977646⟩](https://hal.archives-ouvertes.fr/hal-02977646) - [Paper here](https://hal.archives-ouvertes.fr/hal-02977646/document)

# Our approach
The baseline provided by MediaEval is extended into a two stream network utilising raw RGB and pose data.
This TSPCNN consists of two identical streams, each with five convolutional layers and pool layers with increasing feature sizes and decreasing pool sizes leading into a linear ReLU activation layer.
After the linear layer, there is another linear layer that converts the vector into a 21 dimensional vector for classification (20 different stroke types and non-stroke) respectively a two dimensional vector for detection.
This vector is routed into a softmax layer, fused with the output of the other stream, and passed into a second softmax layer that is used for classification and detection.
The two softmax layers normalize the output of each individual stream before the fusion to minimize vanishing gradients.

![Network](https://github.com/fidsinn/SportTaskME22/blob/master/net.png)

# Results

| Models | Classification | Classification | Classification | Detection | Detection | Detection | Detection |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|  | Train | Validation | Test | Train | Validation | IoU | mAP |
| baseline | - | .813 | .864 | - | - | .515 (.365) | .131 (.118) |
| Pose | 0.995 | .878 | .847 | .862 | .591 | .205 | .046 |
| PRGB | .978 | .813 | .864 | .980 | .834 | .165 | .036 |
| RGB and Pose | 1 | .830 | .872 | .987 | .820 | .331 | .100 |
| RGB and PRGB | .998 | .848 | .873 | .990 | .840 | .349 | .110 |