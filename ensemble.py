"""
Ensembling methods for object detection.
"""
import argparse
import os
import pandas as pd
import numpy as np
import pydicom
import PIL.Image as Image

from tqdm import tqdm
from object_detection.utils import visualization_utils as visutil
from collections import OrderedDict
from calculate_map import parse_submission, calculate_iou

"""
Ensemble - find overlapping boxes of the same class and average their positions
while adding their confidences. Can weigh different detectors with different weights.
No real learning here, although the weights and iou_thresh can be optimized.

Input:
 - dets : List of detections. Each detection is all the output from one detector, and
          should be a list of boxes, where each box should be on the format
          [box_x, box_y, box_w, box_h, class, confidence] where box_x and box_y
          are the center coordinates, box_w and box_h are width and height resp.
          The values should be floats, except the class which should be an integer.

 - iou_thresh: Threshold in terms of IOU where two boxes are considered the same,
               if they also belong to the same class.

 - weights: A list of weights, describing how much more some detectors should
            be trusted compared to others. The list should be as long as the
            number of detections. If this is set to None, then all detectors
            will be considered equally reliable. The sum of weights does not
            necessarily have to be 1.

Output:
    A list of boxes, on the same format as the input. Confidences are in range 0-1.
"""
def ensemble(dets, conf_thresh=0.5, iou_thresh=0.1, weights=None):
    assert(type(iou_thresh) == float)

    ndets = len(dets)

    if weights is None:
        w = 1/float(ndets)
        weights = [w]*ndets
    else:
        assert(len(weights) == ndets)

        s = sum(weights)
        for i in range(0, len(weights)):
            weights[i] /= s

    out = list()
    used = {}

    for idet in range(0,ndets):
        det = dets[idet]
        for box in det:
            if tuple(box) in used:
                continue

            used[tuple(box)] = True
            # Search the other detectors for overlapping box of same class
            found = []
            for iodet in range(0, ndets):
                odet = dets[iodet]

                if odet == det:
                    continue

                bestbox = None
                bestiou = iou_thresh
                for obox in odet:
                    if not tuple(obox) in used:
                        # Not already used
                        if box[4] == obox[4]:
                            # Same class
                            iou = calculate_iou(box[:4], obox[:4])
                            if iou > bestiou:
                                bestiou = iou
                                bestbox = obox

                if not bestbox is None:
                    w = weights[iodet]
                    found.append((bestbox,w))
                    used[tuple(bestbox)] = True

            # Now we've gone through all other detectors
            if len(found) == 0:
                new_box = list(box)
                new_box[5] /= ndets
                if new_box[5] >= conf_thresh:
                    out.append(new_box)
            else:
                allboxes = [(box, weights[idet])]
                allboxes.extend(found)

                xc = 0.0
                yc = 0.0
                bw = 0.0
                bh = 0.0
                conf = 0.0

                wsum = 0.0
                for bb in allboxes:
                    w = bb[1]
                    wsum += w

                    b = bb[0]
                    xc += w*b[0]
                    yc += w*b[1]
                    bw += w*b[2]
                    bh += w*b[3]
                    conf += w*b[5]

                xc /= wsum
                yc /= wsum
                bw /= wsum
                bh /= wsum

                new_box = [xc, yc, bw, bh, box[4], conf]
                if new_box[5] >= conf_thresh:
                    out.append(new_box)
    return out


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dicom_dir', help='path to the directory with DICOM images')
    parser.add_argument('out_images_dir', help='path to the directory for the output images')
    parser.add_argument('ensembled_submission', help='ensembled submission file')
    parser.add_argument('submissions', nargs='+', help='subsmission files to ensemble')
    parser.add_argument('--weights', nargs='+', type=float, help='weight for each submission', default=None)
    parser.add_argument('--conf-thresh', type=float, help='confidence threshold. default 0.5', default=0.5)
    parser.add_argument('--iou-thresh', type=float, help='iou threshold. default 0.1', default=0.1)
    args = parser.parse_args()

    if not args.submissions:
        print('no submissions')
        exit()

    if not os.path.exists(args.out_images_dir):
        os.mkdir(args.out_images_dir)

    submissions = OrderedDict([(s, parse_submission(s)) for s in args.submissions])
    patient_ids = list(submissions[args.submissions[0]].keys())

    submit_dict = {'patientId': [], 'PredictionString': []}
    for pid in tqdm(patient_ids):
        dets = []
        for _, parsed in submissions.items():
            sub_dets = []
            for box, score in zip(parsed[pid]['boxes'], parsed[pid]['scores']):
                sub_dets.append(box + [1, score])
            dets.append(sub_dets)
        ensembled = ensemble(dets, conf_thresh=args.conf_thresh, iou_thresh=args.iou_thresh, weights=args.weights)
        print_boxes = []
        display_boxes = []
        display_classes = []
        display_scores = []
        for e in ensembled:
            x, y, w, h, c, conf = e
            print_boxes.append('{0} {1} {2} {3} {4}'.format(conf, x, y, w, h))
            display_boxes.append([y, x, y + h, x + w])
            display_classes.append(c)
            display_scores.append(conf)
        submit_dict['patientId'].append(pid)
        submit_dict['PredictionString'].append(' '.join(print_boxes))

        im = pydicom.read_file(os.path.join(args.dicom_dir, pid + '.dcm'))
        im = im.pixel_array
        im = np.stack((im, ) * 3, -1)
        if display_boxes:
            visutil.visualize_boxes_and_labels_on_image_array(
                im,
                np.array(display_boxes),
                display_classes,
                display_scores,
                {1: {'id': 1, 'name': 'pneumonia'}},
                use_normalized_coordinates=False,
                min_score_thresh=args.conf_thresh,
            )
        Image.fromarray(im).save(os.path.join(args.out_images_dir, pid + '.jpg'))

    pd.DataFrame(submit_dict).to_csv(args.ensembled_submission, index=False)
