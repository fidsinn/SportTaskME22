<!-- # please respect the structure below-->
*See the [MediaEval 2022 webpage](https://multimediaeval.github.io/editions/2022/) for information on how to register and participate.* <br>
*See the [Sport Task MediaEval 2022 webpage](https://multimediaeval.github.io/editions/2022/tasks/sportsvideo/) for information on the task.*

# Introduction

This is our group's approach to the [Sport Task MediaEval 2022 benchmark competition](https://multimediaeval.github.io/editions/2022/tasks/sportsvideo/).

Running since 2019, this task was focused during the first two years on classification of temporally segmented videos of single table tennis strokes.
Since the third edition of the task, two subtasks have been proposed. The dataset also has been enriched this year with new and more diverse stroke samples.

***Subtask 1 :*** is a classification task: participants are required to build a classification system that automatically labels video segments according to a performed stroke. There are 20 possible stroke classes and an additional non-stroke class.

***Subtask 2 :***  is a more challenging subtask proposed since last year: the goal here is to detect if a stroke has been performed, whatever its classes, and to extract its temporal boundaries. The aim is to be able to distinguish between moments of interest in a game (players performing strokes) from irrelevant moments (picking up the ball, having a break…). This subtask can be a preliminary step for later recognizing a stroke that has been performed.

# Baseline
In order to help participants in their submission, to process videos, annotation files and deep learning techniques, there is a given baseline. The method is simple and is based on the following method using only RGB data:

Pierre-Etienne Martin, Jenny Benois-Pineau, Renaud Péteri, Julien Morlier. 3D attention mechanism for fine-grained classification of table tennis strokes using a Twin Spatio-Temporal Convolutional Neural Networks. 25th International Conference on Pattern Recognition (ICPR2020), Jan 2021, Milano, Italy. [⟨hal-02977646⟩](https://hal.archives-ouvertes.fr/hal-02977646) - [Paper here](https://hal.archives-ouvertes.fr/hal-02977646/document)

We consider 21 classes for the classification task and 2 classes for the detection task. Negative samples are extracted for the detection class and negative proposals are build on the test set. Test is performed with the trimmed proposal (with one window centered or with a sliding window and several post processing approaches) or by running a sliding window on the whole video for the detection task. The lastest output is processed in order to segment in time strokes. Too short strokes are not considered. The model trained on the classification task is also used on the segmentation task without further training on the detection data.

## To cite this work

To cite this work, we invite you to include some previous work. Find the bibTex below. TODO: Update with baseline working note and baseline papers ref.

```
@inproceedings{PeICPR:2020,
  author    = {Pierre{-}Etienne Martin and
              Jenny Benois{-}Pineau and
              Renaud P{\'{e}}teri and
              Julien Morlier},
  title     = {3D attention mechanisms in Twin Spatio-Temporal Convolutional Neural Networks. Application to  action classification in videos of table tennis games.},
  booktitle = {{ICPR}},
  publisher = {{IEEE} Computer Society},
  year      = {2021},
}

@phdthesis{PeThesis:2020,
  author    = {Pierre{-}Etienne Martin},
  title     = {Fine-Grained Action Detection and Classification from Videos with
               Spatio-Temporal Convolutional Neural Networks. Application to Table
               Tennis. (D{\'{e}}tection et classification fines d'actions {\`{a}}
               partir de vid{\'{e}}os par r{\'{e}}seaux de neurones {\`{a}}
               convolutions spatio-temporelles. Application au tennis de table)},
  school    = {University of La Rochelle, France},
  year      = {2020},
  url       = {https://tel.archives-ouvertes.fr/tel-03128769},
  timestamp = {Thu, 25 Feb 2021 08:50:23 +0100},
  biburl    = {https://dblp.org/rec/phd/hal/Martin20a.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}VERSION = {v1},
}
```

