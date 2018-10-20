import pandas as pd
import numpy as np


def read_label_file(label_file):
    df = pd.read_csv(label_file)
    # --- Define lambda to extract coords in list [ymin, xmin, ymax, xmax]
    extract_box = lambda row: [row['y'], row['x'], row['y'] + row['height'], row['x'] + row['width']]

    parsed = {}
    for n, row in df.iterrows():
        # --- Initialize patient entry into parsed
        pid = row['patientId']
        if pid not in parsed:
            parsed[pid] = {
                'dicom': '../input/stage_1_train_images/%s.dcm' % pid,
                'label': row['Target'],
                'boxes': []}

        # --- Add box if opacity is present
        if parsed[pid]['label'] == 1:
            parsed[pid]['boxes'].append(extract_box(row))

    for _, val in parsed.items():
        val['boxes'] = np.array(val['boxes'])
    return parsed
