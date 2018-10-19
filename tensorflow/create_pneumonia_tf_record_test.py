"""Test for create_pneumonia_tf_record.py."""

import os
import datetime
import time

import pandas as pd
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
import tensorflow as tf

import create_pneumonia_tf_record


def write_dicom(pixel_array, filename):
  """
  INPUTS:
  pixel_array: 2D numpy ndarray.  If pixel_array is larger than 2D, errors.
  filename: string name for the output file.
  """

  ## This code block was taken from the output of a MATLAB secondary
  ## capture.  I do not know what the long dotted UIDs mean, but
  ## this code works.
  file_meta = Dataset()
  file_meta.MediaStorageSOPClassUID = 'Secondary Capture Image Storage'
  file_meta.MediaStorageSOPInstanceUID = '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780'
  file_meta.ImplementationClassUID = '1.3.6.1.4.1.9590.100.1.0.100.4.0'
  file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
  ds = FileDataset(filename, {},file_meta = file_meta,preamble=b'\0' * 128)
  ds.Modality = 'WSD'
  ds.ContentDate = str(datetime.date.today()).replace('-','')
  ds.ContentTime = str(time.time()) #milliseconds since the epoch
  ds.StudyInstanceUID =  '1.3.6.1.4.1.9590.100.1.1.124313977412360175234271287472804872093'
  ds.SeriesInstanceUID = '1.3.6.1.4.1.9590.100.1.1.369231118011061003403421859172643143649'
  ds.SOPInstanceUID =    '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780'
  ds.SOPClassUID = 'Secondary Capture Image Storage'
  ds.SecondaryCaptureDeviceManufctur = 'Python 2.7.3'

  ## These are the necessary imaging components of the FileDataset object.
  ds.SamplesPerPixel = 1
  ds.PhotometricInterpretation = 'MONOCHROME2'
  ds.PixelRepresentation = 0
  ds.HighBit = 15
  ds.BitsStored = 16
  ds.BitsAllocated = 16
  ds.SmallestImagePixelValue = b'\\x00\\x00'
  ds.LargestImagePixelValue = b'\\xff\\xff'
  ds.Columns = pixel_array.shape[0]
  ds.Rows = pixel_array.shape[1]
  if pixel_array.dtype != np.uint16:
      pixel_array = pixel_array.astype(np.uint16)
  ds.PixelData = pixel_array.tostring()

  ds.save_as(filename)
  return


class CreateKittiTFRecordTest(tf.test.TestCase):

  def _assertProtoEqual(self, proto_field, expectation):
    """Helper function to assert if a proto field equals some value.

    Args:
      proto_field: The protobuf field to compare.
      expectation: The expected value of the protobuf field.
    """
    proto_list = [p for p in proto_field]
    self.assertListEqual(proto_list, expectation)

  def test_create_tf_example(self):
    width = 1024
    height = 1024
    pid = 'pid0000'
    dicom_file_name = '%s.dcm' % pid
    label = 1
    boxes = [
      {
        'x': 512.0,
        'y': 256.0,
        'width': 512.0,
        'height': 256.0,
      },
      {
        'y': 512.0,
        'x': 256.0,
        'height': 512.0,
        'width': 256.0,
      },
    ]
    image_data = np.random.rand(width, height)
    tmp_dir = self.get_temp_dir()
    save_path = os.path.join(tmp_dir, dicom_file_name)
    write_dicom(image_data, save_path)

    example = create_pneumonia_tf_record.create_tf_example(
      tmp_dir,
      pid,
      label,
      boxes
    )
    self._assertProtoEqual(
      example.features.feature['image/height'].int64_list.value, [height])
    self._assertProtoEqual(
      example.features.feature['image/width'].int64_list.value, [width])
    self._assertProtoEqual(
      example.features.feature['image/filename'].bytes_list.value, [dicom_file_name.encode('utf8')])
    self._assertProtoEqual(
      example.features.feature['image/source_id'].bytes_list.value, [pid.encode('utf8')])
    self._assertProtoEqual(
      example.features.feature['image/format'].bytes_list.value, [b'jpeg'])
    self._assertProtoEqual(
      example.features.feature['image/object/bbox/xmin'].float_list.value,
      [b['x'] / width for b in boxes])
    self._assertProtoEqual(
      example.features.feature['image/object/bbox/xmax'].float_list.value,
      [(b['x'] + b['width']) / width for b in boxes])
    self._assertProtoEqual(
      example.features.feature['image/object/bbox/ymin'].float_list.value,
      [b['y'] / width for b in boxes])
    self._assertProtoEqual(
      example.features.feature['image/object/bbox/ymax'].float_list.value,
      [(b['y'] + b['height']) / height for b in boxes])
    self._assertProtoEqual(
      example.features.feature['image/object/class/text'].bytes_list.value, [b'pneumonia'] * 2)
    self._assertProtoEqual(
      example.features.feature['image/object/class/label'].int64_list.value, [label] * 2)

  def test_parse_labels(self):
    d = {
      'patientId': ['123', '345', '345', '456'],
      'x': [None, 240.0, 120.0, None],
      'y': [None, 235.0, 500.0, None],
      'width': [None, 50.0, 150.0, None],
      'height': [None, 100.0, 150.0, None],
      'Target': [0, 1, 1, 0],
    }
    df = pd.DataFrame.from_dict(d)
    data = create_pneumonia_tf_record.parse_labels(df)
    self.assertEqual(data, {
      '123': {'label': 0, 'boxes': []},
      '345': {'label': 1, 'boxes': [
        {'x': 240.0, 'y': 235.0, 'width': 50.0, 'height': 100.0},
        {'x': 120.0, 'y': 500.0, 'width': 150.0, 'height': 150.0},
      ]},
      '456': {'label': 0, 'boxes': []},
    })


if __name__ == '__main__':
  tf.test.main()
