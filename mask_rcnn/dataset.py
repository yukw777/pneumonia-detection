import os
import glob
import pydicom
import random
import numpy as np
import pandas as pd
import cv2

from mrcnn import utils


def parse_dataset(dicom_dir, anns):
    image_fps = glob.glob(os.path.join(dicom_dir, '*.dcm'))
    image_annotations = {fp: [] for fp in image_fps}
    for _, row in anns.iterrows():
        fp = os.path.join(dicom_dir, row['patientId']+'.dcm')
        image_annotations[fp].append(row)
    return image_fps, image_annotations


class DetectorDataset(utils.Dataset):
    """Dataset class for training pneumonia detection on the RSNA pneumonia dataset.
    """

    def __init__(self, image_fps, image_annotations, orig_height=1024, orig_width=1024):
        super().__init__(self)

        # Add classes
        self.add_class('pneumonia', 1, 'Lung Opacity')

        # add images
        for i, fp in enumerate(image_fps):
            annotations = image_annotations[fp]
            self.add_image('pneumonia', image_id=i, path=fp,
                           annotations=annotations, orig_height=orig_height, orig_width=orig_width)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info['path']
        ds = pydicom.read_file(fp)
        image = ds.pixel_array
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        count = len(annotations)
        if count == 0:
            mask = np.zeros((info['orig_height'], info['orig_width'], 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for i, a in enumerate(annotations):
                if a['Target'] == 1:
                    x = int(a['x'])
                    y = int(a['y'])
                    w = int(a['width'])
                    h = int(a['height'])
                    mask_instance = mask[:, :, i].copy()
                    cv2.rectangle(mask_instance, (x, y), (x+w, y+h), 255, -1)
                    mask[:, :, i] = mask_instance
                    class_ids[i] = 1
        return mask.astype(np.bool), class_ids.astype(np.int32)


def get_train_and_val(dicom_dir, label_file, split=0.04, shuffle=True):
    anns = pd.read_csv(label_file)
    image_fps, image_annotations = parse_dataset(dicom_dir, anns)

    if shuffle:
        sorted(image_fps)
        random.shuffle(image_fps)

    split_index = int((1 - split) * len(image_fps))

    image_fps_train = image_fps[:split_index]
    image_fps_val = image_fps[split_index:]

    dataset_train = DetectorDataset(image_fps_train, image_annotations)
    dataset_train.prepare()

    dataset_val = DetectorDataset(image_fps_val, image_annotations)
    dataset_val.prepare()

    return dataset_train, dataset_val


if __name__ == '__main__':
    import sys
    train, val = get_train_and_val(sys.argv[1], sys.argv[2])

    print('train.image_ids', len(train.image_ids))
    print('val.image_ids', len(val.image_ids))
