# RSNA Pneumonia Detection Challenge

This is a model we created for the RSNA Pneumonia Detection Challenge on Kaggle. It's an ensemble model of various object detection models.

## Usage
### How to train
#### Step 1: Make sure you set up your PYTHONPATH correctly first.
```bash
# from the root directory
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/tensorflow/tf-models/research:`pwd`/tensorflow/tf-models/research/slim:`pwd`/mask_rcnn/Mask_RCNN
```

#### Step 2: Train models and generate submission files.
Please see [darknet](darknet), [tensorflow](tensorflow) and [mask_rcnn](mask_rcnn) directories for the detailed steps to train various models and use them to generate submission files.

#### Step 3: Ensemble.
Use `ensemble.py` to combine the results.
```bash
# this is what we used for the final submission
python ensemble.py
```
