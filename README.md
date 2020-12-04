# object_detection-runescape_rocks
Learning about [TensorFlow's Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) using Runescape as a controlable testing environment. The data pre-processing, model training, and folder structure is based on [TensorFlow's Object Detection](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/) tutorial.

This repo. is mostly a reference for future me, and is offered As-Is with no Guarantees or Warranties. Pull requests welcome. 

# Motivation: 
The MMORPG Old-School Runescape was selected for the basis of experimentation as it provides a low-clutter, highly repeatable, envrionment containing a diverse set of objects that can be used to generate traceable training datasets and perform real-time Model test & evaluation.

# Dataset Preparation
### Image Annotation:
The labeling of class objects was performed using [labelImg](https://github.com/tzutalin/labelImg).

### Dataset Partition:
partition_dataset.py ([source](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#partition-the-dataset)) will split dataset into test/train data based on the -r input ( -r 0.1 => splits 90% train, 10% test).
```
python partition_dataset.py -x -i ./images -r 0.1
```
### Generating Tensorflow .record files for training:
generate_tfrecord.py ([source](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#partition-the-dataset)) converts .xml image annotations into Tensorflow .record binary files for use in model training.
```
python generate_tfrecord.py -x ./images/train -l ./annotations/label_map.pbtxt -o ./annotations/train.record

python generate_tfrecord.py -x ./images/test -l ./annotations/label_map.pbtxt -o ./annotations/test.record
```

### Training the Model:
Select model pipeline.config file from Tensorflow [Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md), for my testing I selected the 640x640 ssd_resnet50_v1_fpn and faster_rcnn_resnet50_v1 (.config's in /models directory). For this dataset and models I get Total Loss ~=1.4 @5,000 steps, and both models train at ~0.8 step/sec on a Nvidia GeForce GTX 1060 6GB CUDA enabled graphics card.

Ensure that any pipeline.config is updated to have the correct: num_classes (10 for this dataset), num_steps, fine_tune_checkpoint (or comment out line), num_steps, and batch_size (depends on computational resources). 


