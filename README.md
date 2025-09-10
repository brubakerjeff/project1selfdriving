# Object detection in an urban environment

In this project, you will learn how to train an object detection model using the [Tensorflow Object Detection API](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html) and [AWS Sagemaker](https://aws.amazon.com/sagemaker/). At the end of this project, you will be able to generate videos such as the one below. 

<p align="center">
    <img src="data/animation.gif" alt="drawing" width="600"/>
</p>

## Installation

Refer to the **Setup Instructions** page in the classroom to setup the Sagemaker Notebook instance required for this project.

>Note: The `conda_tensorflow2_p310` kernel contains most of the required packages for this project. The notebooks contain lines for manual installation when required.

## Usage

This repository contains two notebooks:
* [1_train_model](1_model_training/1_train_model.ipynb): this notebook is used to launch a training job and create tensorboard visualizations. 
* [2_deploy_model](2_run_inference/2_deploy_model.ipynb): this notebook is used to deploy your model, run inference on test data and create a gif similar to the one above.

First, run `1_train_model.ipynb` to train your model. Once the training job is complete, run `2_deploy_model.ipynb` to deploy your model and generate the animation.

Each notebook contains the instructions for running the code, as well the requirements for the writeup. 
>Note: Only the first notebook requires a write up. 

## Useful links
* The Tensorflow Object Detection API tutorial is a great resource to debug your code. This [section](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#configure-the-training-pipeline) in particular will teach you how to edit the `pipeline.config` file to update
your training job.

* [This blog post](https://aws.amazon.com/blogs/machine-learning/training-and-deploying-models-using-tensorflow-2-with-the-object-detection-api-on-amazon-sagemaker/) teaches how to label data, train and deploy a model with the Tensorflow Object Detection API and AWS Sagemaker.
# project1selfdriving
# type: "ssd_efficientnet-b1_bifpn_keras"

# INFO:tensorflow:Finished eval step 100
# I0909 02:28:31.953702 140058636662592 model_lib_v2.py:966] Finished eval step 100
# INFO:tensorflow:Finished eval step 200
# I0909 02:28:39.028575 140058636662592 model_lib_v2.py:966] Finished eval step 200
# INFO:tensorflow:Performing evaluation on 258 images.
# I0909 02:28:43.316850 140058636662592 coco_evaluation.py:293] Performing evaluation on 258 images.
# INFO:tensorflow:Loading and preparing annotation results...
# I0909 02:28:43.321194 140058636662592 coco_tools.py:116] Loading and preparing annotation results...
# INFO:tensorflow:DONE (t=0.01s)
# I0909 02:28:43.333088 140058636662592 coco_tools.py:138] DONE (t=0.01s)
# INFO:tensorflow:Eval metrics at step 2000
# I0909 02:28:51.446383 140058636662592 model_lib_v2.py:1015] Eval metrics at step 2000
# INFO:tensorflow:#011+ DetectionBoxes_Precision/mAP: 0.086089
# I0909 02:28:51.465357 140058636662592 model_lib_v2.py:1018] #011+ DetectionBoxes_Precision/mAP: 0.086089
# INFO:tensorflow:#011+ DetectionBoxes_Precision/mAP@.50IOU: 0.207845
# I0909 02:28:51.466850 140058636662592 model_lib_v2.py:1018] #011+ DetectionBoxes_Precision/mAP@.50IOU: 0.207845
# INFO:tensorflow:#011+ DetectionBoxes_Precision/mAP@.75IOU: 0.064533
# I0909 02:28:51.467874 140058636662592 model_lib_v2.py:1018] #011+ DetectionBoxes_Precision/mAP@.75IOU: 0.064533
# INFO:tensorflow:#011+ DetectionBoxes_Precision/mAP (small): 0.027999
# I0909 02:28:51.468802 140058636662592 model_lib_v2.py:1018] #011+ DetectionBoxes_Precision/mAP (small): 0.027999
# INFO:tensorflow:#011+ DetectionBoxes_Precision/mAP (medium): 0.343860
# I0909 02:28:51.469673 140058636662592 model_lib_v2.py:1018] #011+ DetectionBoxes_Precision/mAP (medium): 0.343860
# INFO:tensorflow:#011+ DetectionBoxes_Precision/mAP (large): 0.306678
# I0909 02:28:51.470530 140058636662592 model_lib_v2.py:1018] #011+ DetectionBoxes_Precision/mAP (large): 0.306678
# INFO:tensorflow:#011+ DetectionBoxes_Recall/AR@1: 0.022020
# I0909 02:28:51.471434 140058636662592 model_lib_v2.py:1018] #011+ DetectionBoxes_Recall/AR@1: 0.022020
# INFO:tensorflow:#011+ DetectionBoxes_Recall/AR@10: 0.095659
# I0909 02:28:51.472312 140058636662592 model_lib_v2.py:1018] #011+ DetectionBoxes_Recall/AR@10: 0.095659
# INFO:tensorflow:#011+ DetectionBoxes_Recall/AR@100: 0.134939
# I0909 02:28:51.473263 140058636662592 model_lib_v2.py:1018] #011+ DetectionBoxes_Recall/AR@100: 0.134939
# INFO:tensorflow:#011+ DetectionBoxes_Recall/AR@100 (small): 0.074425
# I0909 02:28:51.474236 140058636662592 model_lib_v2.py:1018] #011+ DetectionBoxes_Recall/AR@100 (small): 0.074425
# INFO:tensorflow:#011+ DetectionBoxes_Recall/AR@100 (medium): 0.442806
# I0909 02:28:51.475264 140058636662592 model_lib_v2.py:1018] #011+ DetectionBoxes_Recall/AR@100 (medium): 0.442806
# INFO:tensorflow:#011+ DetectionBoxes_Recall/AR@100 (large): 0.528897
# I0909 02:28:51.476346 140058636662592 model_lib_v2.py:1018] #011+ DetectionBoxes_Recall/AR@100 (large): 0.528897
# INFO:tensorflow:#011+ Loss/localization_loss: 0.020034
# I0909 02:28:51.477178 140058636662592 model_lib_v2.py:1018] #011+ Loss/localization_loss: 0.020034
# INFO:tensorflow:#011+ Loss/classification_loss: 0.288811
# I0909 02:28:51.478025 140058636662592 model_lib_v2.py:1018] #011+ Loss/classification_loss: 0.288811
# INFO:tensorflow:#011+ Loss/regularization_loss: 0.030531
# I0909 02:28:51.478828 140058636662592 model_lib_v2.py:1018] #011+ Loss/regularization_loss: 0.030531
# INFO:tensorflow:#011+ Loss/total_loss: 0.339376
# I0909 02:28:51.479627 140058636662592 model_lib_v2.py:1018] #011+ Loss/total_loss: 0.339376
# INFO:tensorflow:Waiting for new checkpoint at /opt/training
# I0909 02:32:55.568623 140058636662592 checkpoint_utils.py:168] Waiting for new checkpoint at /opt/# training
# INFO:tensorflow:Timed-out waiting for a checkpoint.
# I0909 02:33:04.583636 140058636662592 checkpoint_utils.py:231] Timed-out waiting for a checkpoint.
# creating index...
# index created!
# creating index...
# index created!
# Running per image evaluation...
# Evaluate annotation type *bbox*
# DONE (t=7.90s).
# Accumulating evaluation results...

# DONE (t=0.17s).

#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.086
 # Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.208
 # Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.065
 # Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.028
 # Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.344
 # Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.307
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.022
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.096
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.135
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.074
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.443
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.529