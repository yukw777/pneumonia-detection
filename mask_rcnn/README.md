# Mask RCNN

## How to train
### Step 1: Set the correct PYTHONPATH
See the main [README.md](../README.md)

### Step 2: Run the training script
Use `train.py` to train a model. Configs we used are [here](models). Download a pretrained model to speed up the training. We used the [pretrained model](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5) for COCO. One thing to note is that this script splits the training data into train and validation sets in memory.
```bash
python train.py models/resnet50/config.ini \
    /path/to/train/dicoms \
    /path/to/train/label \
    /output/dir \
    --coco /path/to/coco/model.h5
```

### Step 3: Evaluate and test
Use `eval.py` to evaluate or test the model.
```bash
# evaluate
python eval.py \
    models/resnet50/config.ini \
    0.1 \
    /path/to/dicom/images \
    /path/to/trained/model \
    /path/to/output/dir \
    --label-file /path/to/label/file \
    --pick-500

# test
# this is what we used for our submission
python eval.py \
    models/resnet50/config.ini \
    0.95 \
    /path/to/dicom/images \
    /path/to/trained/model \
    /path/to/output/dir \
    --submission-file /path/to/submission/file
python eval.py \
    models/resnet101/config.ini \
    0.98 \
    /path/to/dicom/images \
    /path/to/trained/model \
    /path/to/output/dir \
    --submission-file /path/to/submission/file
```
