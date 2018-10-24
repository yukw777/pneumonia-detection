# RSNA Pneumonia Detection Challenge

This is a model we created for the RSNA Pneumonia Detection Challenge on Kaggle. It's an ensemble model of various object detection models.

## Usage
### How to train
#### Step 1: Check out the submodules
```bash
git submodule update --init
```

#### Step 2: Make sure you set up your PYTHONPATH correctly first.
```bash
# from the root directory
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/tensorflow/tf-models/research:`pwd`/tensorflow/tf-models/research/slim:`pwd`/mask_rcnn/Mask_RCNN
```

#### Step 3: Train models and generate submission files.
Please see [darknet](darknet), [tensorflow](tensorflow) and [mask_rcnn](mask_rcnn) directories for the detailed steps to train various models and use them to generate submission files.

#### Step 4: Ensemble.
Use `ensemble.py` to combine the results.
```bash
# this is what we used for the final submission
# First generate 7 submission files for all the models
# then run the following to generate an ensembled submission file
python ensemble.py
    <ensembled.csv> \
    <faster_rcnn_resnet101 submission> \
    <faster_rcnn_resnet50 submission> \
    <faster_rcnn_inception v2 submission> \
    <mask_rcnn_resnet101 submission> \
    <mask_rcnn_resnet50 submission> \
    <ssd_mobilenet_v2 submission> \
    <darknet_yolov3 submission> \
    --iou-thresh 0.01 \
    --conf-thresh 0.4 \
    --weights 0.204 0.18 0.168 0.16 0.145 0.12 0.136
```
