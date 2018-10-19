import argparse
import csv
import os


def generate_label(row, width, height):
    x_c = float(row['x']) + float(row['width']) / 2
    x_c /= width
    y_c = float(row['y']) + float(row['height']) / 2
    y_c /= height
    w = float(row['width']) / width
    h = float(row['height']) / height
    return ['0', x_c, y_c, w, h]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Turn the labels from Kaggle to darknet')
    parser.add_argument('labels', help='labels file from Kaggle')
    parser.add_argument('darknet', help='darknet labels folder')
    parser.add_argument('-w', '--width', help='image width (default: 1024.0)',
                        type=float, default=1024.0)
    parser.add_argument('-e', '--height', help='image height (default: 1024.0)',
                        type=float, default=1024.0)

    args = parser.parse_args()

    if not os.path.exists(args.darknet):
        os.makedirs(args.darknet)

    with open(args.labels, newline='') as labels:
        reader = csv.DictReader(labels)
        for row in reader:
            label_path = os.path.join(args.darknet, row['patientId'] + '.txt')
            with open(label_path, 'a', newline='') as darknet_label:
                if row['Target'] == '1':
                    writer = csv.writer(darknet_label, delimiter=' ')
                    writer.writerow(generate_label(
                        row, args.width, args.height))
