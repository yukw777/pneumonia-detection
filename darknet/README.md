# Darknet
## How to train

### Step 1: Convert DICOM to JPG
The data given to us by RSNA is DICOM, but Darknet only accepts JPG. Also, Darknet keeps track of images via text files. The following commands will help you generate everything that's required to train a Darknet model from the RSNA DICOM data set. Most Linux distributions come with ImageMagick installed. Let's use it to convert DICOMs to JPGs.
```bash
# we need to set TrueColor b/c the DICOMs we have are grayscale,
# but, like most object detection algorithms, Darknet expects
# colored images
mogrify -format jpg -type TrueColor -path /dir/to/save/jpgs /path/to/dicoms
```

### Step 2: Split data into train and valid sets
Now that we have JPGs, let's split them into the train set and the validation set.
```bash
# get all the names of JPGs, shuffle them and save them into a file
find /dir/with/jpgs | grep jpg | shuf > files.txt
# Take the first 5000 as the validation set, the rest as train
(head -5000 > valid.txt; cat > train.txt) < files.txt
```

### Step 3: Create the labels
Use the following Python script to generate labels for the train and validation sets.
```bash
python darknet/generate_darknet_labels.py <labels.csv> /dir/to/save/labels/files
```

### Step 4: Create config files
Darknet requires three config files to train. Samples we used are in this repo. Please see the [Darknet official website](https://pjreddie.com/darknet/yolo/) for more details.
1. [data file](cfg/yolov3.data)
1. [name file](cfg/yolov3.names)
1. config file: [train](cfg/yolov3-train.cfg), [test](cfg/yolov3-test.cfg)

### Step 5: Train!
You must have the darknet binary compiled. Please see the official website for [instructions](https://pjreddie.com/darknet/install/). In order to speed up the training, we used the [pretrained model](https://pjreddie.com/media/files/darknet53.conv.74) from the official Darknet website.
```bash
# single gpu
./darknet detector cfg/yolov3.data cfg/yolov3-train.cfg /path/to/pretrained | tee train.log

# multi gpu
./darknet detector cfg/yolov3.data cfg/yolov3-train.cfg /path/to/pretrained -gpus 0,1,2,3 | tee train.log

# loss plot
python loss_plot.py train.log
```

## How to evaluate and test
Use `darknet_test.py` to evaluate and test the model. It requires `libdarknet.so` which is generated when you compile the Darknet binary.
```bash
# convert test DICOMs to JPGs
mogrify -format jpg -type TrueColor -path /dir/to/save/jpgs /path/to/test/dicoms

# generate a test data file
find /dir/to/save/jpgs | grep jpg > test.txt

# evaluate. produces a set of images with
# top 3 detected boxes and groundtruth boxes
python darknet_test.py cfg/yolov3-test.cfg <weight_file> cfg/yolov3.data test.txt <submission.csv> -t 0.01 --label-file <label-file.csv>

# test
# this is what we used for our submission
python darknet_test.py cfg/yolov3-test.cfg <weight_file> cfg/yolov3.data test.txt <submission.csv> -t 0.03
```
