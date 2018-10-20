import numpy as np
import pandas as pd
import argparse


def calculate_iou(box1, box2):
    x11, y11, w1, h1 = box1
    x21, y21, w2, h2 = box2
    assert w1 * h1 > 0
    assert w2 * h2 > 0
    x12, y12 = x11 + w1, y11 + h1
    x22, y22 = x21 + w2, y21 + h2

    area1, area2 = w1 * h1, w2 * h2
    xi1, yi1, xi2, yi2 = max([x11, x21]), max([y11, y21]), min([x12, x22]), min([y12, y22])

    if xi2 <= xi1 or yi2 <= yi1:
        return 0

    intersect = (xi2-xi1) * (yi2-yi1)
    union = area1 + area2 - intersect
    return intersect / union


def map_iou(boxes_true, boxes_pred, scores, thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]):
    """
    Mean average precision at differnet intersection over union (IoU) threshold

    input:
        boxes_true: Mx4 numpy array of ground true bounding boxes of one image.
                    bbox format: (x1, y1, w, h)
        boxes_pred: Nx4 numpy array of predicted bounding boxes of one image.
                    bbox format: (x1, y1, w, h)
        scores:     length N numpy array of scores associated with predicted bboxes
        thresholds: IoU shresholds to evaluate mean average precision on
    output:
        map: mean average precision of the image
    """

    # According to the introduction, images with no ground truth bboxes will not be
    # included in the map score unless there is a false positive detection (?)

    # return None if both are empty, don't count the image in final evaluation (?)
    if len(boxes_true) == 0 and len(boxes_pred) == 0:
        return None

    assert boxes_true.shape[1] == 4 or boxes_pred.shape[1] == 4, "boxes should be 2D arrays with shape[1]=4"
    if len(boxes_pred):
        assert len(scores) == len(boxes_pred), "boxes_pred and scores should be same length"
        # sort boxes_pred by scores in decreasing order
        boxes_pred = boxes_pred[np.argsort(scores)[::-1], :]

    map_total = 0

    # loop over thresholds
    for t in thresholds:
        matched_bt = set()
        tp, fn = 0, 0
        for i, bt in enumerate(boxes_true):
            matched = False
            for j, bp in enumerate(boxes_pred):
                miou = calculate_iou(bt, bp)
                if miou >= t and not matched and j not in matched_bt:
                    matched = True
                    tp += 1 # bt is matched for the first time, count as TP
                    matched_bt.add(j)
            if not matched:
                fn += 1 # bt has no match, count as FN

        fp = len(boxes_pred) - len(matched_bt) # FP is the bp that not matched to any bt
        m = tp / (tp + fn + fp)
        map_total += m

    return map_total / len(thresholds)


def parse_labels(df):
    """
    Method to read a CSV file (Pandas dataframe) and parse the
    data into the following nested dictionary:

      parsed = {
        'patientId-00': list of box(es)
        'patientId-01': list of box(es),
        ...
      }

    """
    # --- Define lambda to extract coords in list [y, x, height, width]
    extract_box = lambda row: [row['y'], row['x'], row['height'], row['width']]

    parsed = {}
    for _, row in df.iterrows():
        # --- Initialize patient entry into parsed
        pid = row['patientId']
        if pid not in parsed:
            parsed[pid] = []

        # --- Add box if opacity is present
        if row['Target'] == 1:
            parsed[pid].append(extract_box(row))

    return parsed


def parse_pred_string(s):
    parts = s.split()
    for i in range(0, len(parts), 5):
        subparts = parts[i: i + 5]
        yield {
            'score': float(subparts[0]),
            'box': [float(p) for p in subparts[1:]],
        }


def parse_submission(submission):
    """
    Parse a pandas dataframe that contains data from a submission file.

      parsed = {
        'patientId-00': {
            'boxes': list of box(es),
            'scores': list of score(s),
        },
        'patientId-01': {
            'boxes': list of box(es),
            'scores': list of score(s),
        },
        ...
      }

    """
    df = pd.read_csv(
        submission,
        dtype={'PredictionString': str},
        keep_default_na=False)
    parsed = {}
    for _, row in df.iterrows():
        pid = row['patientId']
        boxes = list(parse_pred_string(row['PredictionString']))
        parsed[pid] = {
            'scores': [b['score'] for b in boxes],
            'boxes': [b['box'] for b in boxes],
        }

    return parsed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('labels')
    parser.add_argument('submission')
    args = parser.parse_args()

    labels = parse_labels(pd.read_csv(args.labels))
    submission = parse_submission(args.submission)

    count = 0
    total = 0.0
    for  k in submission.keys():
        print('label', labels[k], 'submission', submission[k])
        if labels[k] == []:
            if submission[k] == []:
                continue
            else:
                count += 1
        else:
            iou = map_iou(
                np.array(labels[k]),
                np.array(submission[k]['boxes']),
                np.array(submission[k]['scores'])
            )
            print('calculated iou for this box:', iou)
            total += iou
            count += 1

    print('Final mAP:', total/count)
