# ==============================================================================
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4" # "0,1,2,3,4" for multiple
# ==============================================================================

import ctypes
import math
import random
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

from utils import read_label_file
from object_detection.utils import visualization_utils as visutil


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float),
                ("y", ctypes.c_float),
                ("w", ctypes.c_float),
                ("h", ctypes.c_float)]

class DETECTION(ctypes.Structure):
    _fields_ = [("bbox", BOX),
                ("classes", ctypes.c_int),
                ("prob", ctypes.POINTER(ctypes.c_float)),
                ("mask", ctypes.POINTER(ctypes.c_float)),
                ("objectness", ctypes.c_float),
                ("sort_class", ctypes.c_int)]


class IMAGE(ctypes.Structure):
    _fields_ = [("w", ctypes.c_int),
                ("h", ctypes.c_int),
                ("c", ctypes.c_int),
                ("data", ctypes.POINTER(ctypes.c_float))]

class METADATA(ctypes.Structure):
    _fields_ = [("classes", ctypes.c_int),
                ("names", ctypes.POINTER(ctypes.c_char_p))]


# ==============================================================================
#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
darknet_lib_path = os.path.join(os.getcwd(), "libdarknet.so")
lib = ctypes.CDLL(darknet_lib_path, ctypes.RTLD_GLOBAL)
# ==============================================================================
lib.network_width.argtypes = [ctypes.c_void_p]
lib.network_width.restype = ctypes.c_int
lib.network_height.argtypes = [ctypes.c_void_p]
lib.network_height.restype = ctypes.c_int

predict = lib.network_predict
predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
predict.restype = ctypes.POINTER(ctypes.c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [ctypes.c_int]

make_image = lib.make_image
make_image.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int)
]
get_network_boxes.restype = ctypes.POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [ctypes.c_void_p]
make_network_boxes.restype = ctypes.POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [ctypes.POINTER(DETECTION), ctypes.c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]

network_predict = lib.network_predict
network_predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [ctypes.c_void_p]

load_net = lib.load_network
load_net.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
load_net.restype = ctypes.c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [ctypes.POINTER(DETECTION), ctypes.c_int, ctypes.c_int, ctypes.c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [ctypes.POINTER(DETECTION), ctypes.c_int, ctypes.c_int, ctypes.c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, ctypes.c_int, ctypes.c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [ctypes.c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [ctypes.c_void_p, IMAGE]
predict_image.restype = ctypes.POINTER(ctypes.c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = ctypes.c_int(0)
    pnum = ctypes.pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms):
        do_nms_obj(dets, num, meta.classes, nms)

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('weight_file')
    parser.add_argument('data_file')
    parser.add_argument('test_images_file')
    parser.add_argument('submission_file')
    parser.add_argument('images_out_dir')
    parser.add_argument('--label-file')
    parser.add_argument('-g', '--gpu-index', type=int, default=0)
    parser.add_argument('-t', '--threshold', type=float, default=0.2)
    args = parser.parse_args()

    if not os.path.exists(args.images_out_dir):
        os.mkdir(args.images_out_dir)

    labels = None
    if args.label_file:
        labels = read_label_file(args.label_file)

    net = load_net(args.config_file.encode(), args.weight_file.encode(), args.gpu_index)
    meta = load_meta(args.data_file.encode())

    submit_dict = {'patientId': [], 'PredictionString': []}

    with open(args.test_images_file, 'r') as test_images_file:
        for line in tqdm(test_images_file):
            patient_id = os.path.split(line.strip())[1].split('.')[0]

            # read the image
            im = np.array(Image.open(line.strip()))

            infer_result = detect(
                net, meta, line.strip().encode(), thresh=args.threshold)

            boxes = []
            display_boxes = []
            display_scores = []
            for e in infer_result:
                confidence = e[1]
                display_scores.append(confidence)
                w = e[2][2]
                h = e[2][3]
                x = e[2][0]-w/2
                y = e[2][1]-h/2
                display_boxes.append([y, x, y + h, x + w])
                boxes.append('{0} {1} {2} {3} {4}'.format(confidence, x, y, w, h))

            submit_dict['patientId'].append(patient_id)
            submit_dict['PredictionString'].append(' '.join(boxes))

            if infer_result:
                visutil.visualize_boxes_and_labels_on_image_array(
                    im,
                    np.array(display_boxes),
                    [1] * len(display_boxes),
                    display_scores,
                    {1: {'id': 1, 'name': 'pneumonia'}},
                    use_normalized_coordinates=False,
                    max_boxes_to_draw=3,
                    min_score_thresh=args.threshold,
                )
            if labels:
                visutil.visualize_boxes_and_labels_on_image_array(
                    im,
                    labels[patient_id]['boxes'],
                    [labels[patient_id]['label']] * len(labels[patient_id]['boxes']),
                    None,
                    {1: {'id': 1, 'name': 'pneumonia'}},
                    use_normalized_coordinates=False,
                )
            im = Image.fromarray(im)
            im.save(os.path.join(args.images_out_dir, patient_id + '.jpg'))

    pd.DataFrame(submit_dict).to_csv(args.submission_file, index=False)
