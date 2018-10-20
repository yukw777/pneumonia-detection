# Google Object Detection API

We used Google's Object Detection API to train various object detection models: SSD Mobilenet v2, Faster RCNN Inception v2, Faster RCNN Resnet50 and Faster RCNN Resnet101.

## How to set up Google's Object Detection API
The detailed instructions can be found on the [official website](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md), but here is the overview:

### Step 1: Install the required packages
Simply run `pip install -r requirements.txt` from the root directory.

### Step 2: Instll COCO API
```bash
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools <root_dir>/tensorflow/tf-models/research/
```

### Step 3: Compile protobuf
```bash
# from <root_dir>/tensorflow/tf-models/research/
protoc object_detection/protos/*.proto --python_out=.
```

### Step 4: Set PYTHONPATH
Follow the instructions on the main [README.md](../README.md).

## How to train

### Step 1: Convert DICOMs to Tensorflow Records
Use `create_pneumonia_tf_record.py` to convert DICOM images to Tensorflow Records.
```bash
python create_pneumonia_tf_record.py --dicom_dir /dir/to/DICOMs --eval_perct 0.2 --label_file /path/to/label/file --output_dir data/records
```

### Step 2: Download pretrained models
You can download pretrained models from Google's Object Detection API [Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). We used the following:
- [SSD Mobilenet v2](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)
- [Faster RCNN Inception v2](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)
- [Faster RCNN Resnet50](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz)
- [Faster RCNN Resnet101](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz)

The sample config files expect the pretrained models to be in the model file. For example, for SSD Mobilenet v2, place the pretrained model [here](models/ssd_mobilenet_v2).

### Step 3: Train
Use Google's Object Detection API to train the models. The config files are [here](models).
```bash
# train SSD Mobilenet v2
python tf-models/research/object_detection/model_main.py \
    --pipeline_config_paath models/ssd_mobilenet_v2/ssd_mobilenet_v2_pneumonia.config \
    --model_dir /path/to/output
```

### Step 4: Export the trained model for inference
Once you're done with training, you need to export the model so that you can use it for inference. You can also override some of the model config values such as the max suppression confidence threshold and IOU threshold.

```bash
# without config override
python tf-models/research/object_detection/export_inference_graph.py \
    --pipeline_config_path /path/to/trained/pipeline.config \
    --trained_checkpoint_prefix /path/to/trained/model.ckpt-<step_number> \
    --output_directory /output/dir

# with config override for faster rcnn
python tf-models/research/object_detection/export_inference_graph.py \
    --pipeline_config_path /path/to/trained/pipeline.config \
    --trained_checkpoint_prefix /path/to/trained/model.ckpt-<step_number> \
    --output_directory /output/dir \
    --config_override "
    model {
        faster_rcnn {
            second_stage_post_processing {
                batch_non_max_suppression {
                    score_threshold: 0.3
                    iou_threshold: 0.1
                    max_detection_per_class: 10
                    max_total_detection: 10
                }
            }
        }
    }"
```

### Step 5: Evaluate
Use `model_eval.py` to generate evaluation images with detected boxes to see how your model performs.
```bash
python model_eval.py \
    --label_map data/pneumonia_label_map.pbtxt \
    --inference_graph /path/to/exported/frozen_inference_graph.pb \
    --input_tfrecord_pattern "data/records/pneumonia_eval.record-?????-of-00010" \
    --output_images_dir /path/to/output
```

### Step 6: Test
Use `model_infer.py` to test the model on new DICOM images and generate a submission file.
```bash
python model_infer.py \
    /path/to/exported/frozen_inference_graph.pb \
    data/pneumonia_label_map.pbtxt \
    /path/to/test/dicoms \
    /path/to/submission/file \
    /path/to/output/image/dir
```
