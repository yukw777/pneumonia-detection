import argparse
import os
import glob
import pydicom
import numpy as np
import random
import pandas as pd
import mrcnn.model as modellib
import PIL.Image as Image

from object_detection.utils import visualization_utils as visutil
from mrcnn import utils
from config import from_config_file
from tqdm import tqdm
from utils import read_label_file


# Make predictions on test images, write out sample submission
def predict(image_fps, config, min_conf, submission_file=None, label_file=None, eval_dir=None):
    labels = None
    if label_file:
        labels = read_label_file(label_file)
    submit_dict = None
    if submission_file:
        submit_dict = {'patientId': [], 'PredictionString': []}
    for image_id in tqdm(image_fps):
        ds = pydicom.read_file(image_id)
        original_image = ds.pixel_array
        resize_factor = original_image.shape[0] / config.IMAGE_MIN_DIM

        # Convert to RGB for consistency.
        original_image = np.stack((original_image,) * 3, -1)
        image, _, _, _, _ = utils.resize_image(
            original_image,
            min_dim=config.IMAGE_MIN_DIM,
            min_scale=config.IMAGE_MIN_SCALE,
            max_dim=config.IMAGE_MAX_DIM,
            mode=config.IMAGE_RESIZE_MODE)

        patient_id = os.path.splitext(os.path.basename(image_id))[0]

        results = model.detect([image])
        r = results[0]

        assert( len(r['rois']) == len(r['class_ids']) == len(r['scores']) )
        if submit_dict:
            submit_dict['patientId'].append(patient_id)
        boxes = []
        for i in range(len(r['rois'])):
            if r['scores'][i] > min_conf:
                x1 = r['rois'][i][1]
                y1 = r['rois'][i][0]
                width = (r['rois'][i][3] - x1)
                height = (r['rois'][i][2] - y1)
                boxes.append('{0} {1} {2} {3} {4}'.format(
                    r['scores'][i],
                    x1 * resize_factor,
                    y1 * resize_factor,
                    width * resize_factor,
                    height * resize_factor))
        if submit_dict:
            submit_dict['PredictionString'].append(' '.join(boxes))

        # draw predicted boxes
        if len(r['rois']) != 0:
            visutil.visualize_boxes_and_labels_on_image_array(
                original_image,
                np.array(r['rois']) * resize_factor,
                r['class_ids'],
                r['scores'],
                {1: {'id': 1, 'name': 'pneumonia'}},
                use_normalized_coordinates=False,
                max_boxes_to_draw=3,
                min_score_thresh=min_conf,
            )

        # draw ground truth boxes
        if labels:
            visutil.visualize_boxes_and_labels_on_image_array(
                original_image,
                labels[patient_id]['boxes'],
                [labels[patient_id]['label']] * len(labels[patient_id]['boxes']),
                None,
                {1: {'id': 1, 'name': 'pneumonia'}},
                use_normalized_coordinates=False,
            )

        im = Image.fromarray(original_image)
        im.save(os.path.join(eval_dir, patient_id + '.jpg'))

    if submit_dict:
        pd.DataFrame(submit_dict).to_csv(submission_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='path to train config file')
    parser.add_argument('threshold', help='min threshold for detected boxes', type=float)
    parser.add_argument('dicom_dir', help='path to directory with eval dicom files')
    parser.add_argument('model', help='path to the pretrained model')
    parser.add_argument('eval_dir', help='path to directory with evaluated images')
    parser.add_argument('--label-file', help='path to label file')
    parser.add_argument('--submission-file', help='path to output submission file')
    parser.add_argument('--pick-500', help='pick 500 random images', action='store_true')
    args = parser.parse_args()

    # create the eval dir
    if not os.path.exists(args.eval_dir):
        os.mkdir(args.eval_dir)

    # Set inference config
    config = from_config_file(args.config)
    config.GPU_COUNT = 1
    config.IMAGES_PER_GPU = 1
    config.DETECTION_MAX_INSTANCES = 10
    config.__init__()

    # Recreate the model in inference mode
    print('Loading weights from ' + args.model)
    model = modellib.MaskRCNN(
        mode='inference',
        config=config,
        model_dir=os.path.dirname(args.model)
    )
    model.load_weights(args.model, by_name=True)
    print('Weights loaded')

    # Get filenames of test dataset DICOM images
    test_image_fps = glob.glob(os.path.join(args.dicom_dir, '*.dcm'))

    if args.pick_500:
        sorted(test_image_fps)
        random.shuffle(test_image_fps)
        test_image_fps = test_image_fps[:500]

    predict(
        test_image_fps,
        config,
        args.threshold,
        submission_file=args.submission_file,
        label_file=args.label_file,
        eval_dir=args.eval_dir
    )
