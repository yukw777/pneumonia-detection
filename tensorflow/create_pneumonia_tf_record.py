import tensorflow as tf
import contextlib2
import pandas as pd
import pydicom
import os
import random

from object_detection.utils import dataset_util
from object_detection.dataset_tools import tf_record_creation_util


flags = tf.app.flags
flags.DEFINE_string('dicom_dir', '', 'DICOM data directory.')
flags.DEFINE_string('label_file', '', 'label csv file')
flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')
flags.DEFINE_float('eval_perct', 0.2, 'What percentage of data is eval')
FLAGS = flags.FLAGS


def create_tf_example(dicom_dir, pid, label, boxes):
  '''Converts DICOM image and annotations to a tf.Example proto.

  Args:
    dicom_dir: path to directory with DICOM images
    pid: patient id
    label: 1 if pneumonia, 0 otherwise
    boxes: list of bounding boxes for opacities
  Returns:
    example: The converted tf.Example
  '''

  # Read the DICOM image
  filename = '%s.dcm' % pid
  dcm = pydicom.read_file(os.path.join(dicom_dir, filename))

  # extract image height and width
  width, height = dcm.pixel_array.shape

  # convert from gray scale to 3-channel RGB
  encoded_image_data = pydicom.encaps.defragment_data(dcm.PixelData)
  image_format = b'jpeg'

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalizedjj bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)
  for box in boxes:
    classes_text.append(b'pneumonia')
    classes.append(label)
    xmins.append(box['x'] / width)
    xmaxs.append((box['x'] + box['width']) / width)
    ymins.append(box['y'] / height)
    ymaxs.append((box['y'] + box['height']) / height)


  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(pid.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


def parse_labels(df):
  '''Converts a pandas dataframe to a dictionary keyed by patient id.

  Args:
    df: pandas dataframe from the labels file.
  Returns:
    a dictionary with patient label data.
    {
      'patientId-00': {
        'label': either 0 or 1 for normal or pneumonia,
        'boxes': list of box(es)
      }, ...
    }
  '''
  parsed = {}
  for _, row in df.iterrows():
    pid = row['patientId']
    if pid not in parsed:
      parsed[pid] = {
        'label': row['Target'],
        'boxes': [],
      }
    if parsed[pid]['label'] == 1:
      parsed[pid]['boxes'].append({
        'x': row['x'],
        'y': row['y'],
        'width': row['width'],
        'height': row['height'],
      })

  return parsed


def _create_tf_record_from_rsna_set(parsed, pids, num_shards, record_name):
  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
      tf_record_close_stack,
      os.path.join(FLAGS.output_dir, record_name),
      num_shards
    )
    for index, pid in enumerate(pids):
      data = parsed[pid]
      tf_example = create_tf_example(
        FLAGS.dicom_dir, pid, data['label'], data['boxes'])
      output_shard_index = index % num_shards
      output_tfrecords[output_shard_index].write(tf_example.SerializeToString())


def main(_):
  assert FLAGS.dicom_dir, '`dicom_dir` missing.'
  assert FLAGS.label_file, '`label_file` missing.'

  if not os.path.isdir(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)

  df = pd.read_csv(FLAGS.label_file)
  parsed = parse_labels(df)
  pids = list(parsed.keys())
  random.shuffle(pids)
  split = int(len(pids) * FLAGS.eval_perct)
  eval_pids = pids[:split]
  train_pids = pids[split:]
  num_shards=10

  _create_tf_record_from_rsna_set(parsed, train_pids, num_shards, 'pneumonia_train.record')
  _create_tf_record_from_rsna_set(parsed, eval_pids, num_shards, 'pneumonia_eval.record')

if __name__ == '__main__':
  tf.app.run()
