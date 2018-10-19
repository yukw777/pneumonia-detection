# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Infers detections on TFRecords of TFExamples given an inference graph.

Example usage:
  ./model_eval.py \
    --input_tfrecord_pattern=/path/to/input/tfrecord1* \
    --output_images_dir=/path/to/output/detections/images \
    --inference_graph=/path/to/frozen_weights_inference_graph.pb \
    --label_map=/path/to/label_map

The output is a collection of jpeg images. Each image has the ground truth boxes
drawn, as well as the inferred bounding boxes.

The input and output nodes of the inference graph are expected to have the same
types, shapes, and semantics, as the input and output nodes of graphs produced
by export_inference_graph.py, when run with --input_type=image_tensor.
"""

import glob
import os
import itertools
import tensorflow as tf
import numpy as np
from object_detection.inference import detection_inference
from object_detection.core import standard_fields
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from PIL import Image
from io import BytesIO

tf.flags.DEFINE_string('input_tfrecord_pattern', None,
                       'glob pattern for input tfrecords')
tf.flags.DEFINE_string('output_images_dir', None,
                       'Path to the output images.')
tf.flags.DEFINE_string('inference_graph', None,
                       'Path to the inference graph with embedded weights.')
tf.flags.DEFINE_string('label_map', None,
                       'Path to the label map file.')
FLAGS = tf.flags.FLAGS


def get_image_array_from_example(tf_example):
  encoded_jpg = tf_example.features.feature[standard_fields.TfExampleFields.image_encoded].bytes_list.value[0]
  image = Image.open(BytesIO(encoded_jpg))
  image_np = np.array(image)
  return np.stack([image_np] * 3, axis=2)


def draw_bounding_boxes_from_example(image_np, tf_example, category_index):
  # detected bounding boxes
  d_xmin = tf_example.features.feature[standard_fields.TfExampleFields.detection_bbox_xmin].float_list.value
  d_xmax = tf_example.features.feature[standard_fields.TfExampleFields.detection_bbox_xmax].float_list.value
  d_ymin = tf_example.features.feature[standard_fields.TfExampleFields.detection_bbox_ymin].float_list.value
  d_ymax = tf_example.features.feature[standard_fields.TfExampleFields.detection_bbox_ymax].float_list.value
  d_classes = tf_example.features.feature[standard_fields.TfExampleFields.detection_class_label].int64_list.value
  d_scores = tf_example.features.feature[standard_fields.TfExampleFields.detection_score].float_list.value
  vis_util.visualize_boxes_and_labels_on_image_array(
    image_np,
    np.stack([d_ymin, d_xmin, d_ymax, d_xmax], axis=1),
    d_classes,
    d_scores,
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=3,
    min_score_thresh=0.0001
  )

  # ground truth bounding boxes
  g_xmin = tf_example.features.feature[standard_fields.TfExampleFields.object_bbox_xmin].float_list.value
  g_xmax = tf_example.features.feature[standard_fields.TfExampleFields.object_bbox_xmax].float_list.value
  g_ymin = tf_example.features.feature[standard_fields.TfExampleFields.object_bbox_ymin].float_list.value
  g_ymax = tf_example.features.feature[standard_fields.TfExampleFields.object_bbox_ymax].float_list.value
  g_classes = tf_example.features.feature[standard_fields.TfExampleFields.object_class_label].int64_list.value
  vis_util.visualize_boxes_and_labels_on_image_array(
    image_np,
    np.stack([g_ymin, g_xmin, g_ymax, g_xmax], axis=1),
    g_classes,
    None,
    category_index,
    use_normalized_coordinates=True,
  )


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  required_flags = ['input_tfrecord_pattern', 'output_images_dir',
                    'inference_graph', 'label_map']
  for flag_name in required_flags:
    if not getattr(FLAGS, flag_name):
      raise ValueError('Flag --{} is required'.format(flag_name))

  # load the categories
  category_index = label_map_util.create_category_index_from_labelmap(FLAGS.label_map, use_display_name=True)

  # create the outputdir if it doesn't exist already
  if not os.path.exists(FLAGS.output_images_dir):
    os.mkdir(FLAGS.output_images_dir)

  with tf.Session() as sess:
    input_tfrecord_paths = glob.glob(FLAGS.input_tfrecord_pattern)
    tf.logging.info('Reading input from %d files', len(input_tfrecord_paths))
    serialized_example_tensor, image_tensor = detection_inference.build_input(
        input_tfrecord_paths)
    tf.logging.info('Reading graph and building model...')
    (detected_boxes_tensor, detected_scores_tensor,
     detected_labels_tensor) = detection_inference.build_inference_graph(
         image_tensor, FLAGS.inference_graph)

    tf.logging.info('Running inference and writing output to {}'.format(
        FLAGS.output_images_dir))
    sess.run(tf.local_variables_initializer())
    tf.train.start_queue_runners()
    try:
      for counter in itertools.count():
        tf.logging.log_every_n(tf.logging.INFO, 'Processed %d images...', 10, counter)
        tf_example = detection_inference.infer_detections_and_add_to_example(
            serialized_example_tensor, detected_boxes_tensor,
            detected_scores_tensor, detected_labels_tensor,
            False)
        image_np = get_image_array_from_example(tf_example)
        draw_bounding_boxes_from_example(image_np, tf_example, category_index)
        im = Image.fromarray(image_np)
        pid = tf_example.features.feature[standard_fields.TfExampleFields.source_id].bytes_list.value[0].decode()
        im.save(os.path.join(FLAGS.output_images_dir, pid + '.jpg'))
    except tf.errors.OutOfRangeError:
      tf.logging.info('Finished processing records')


if __name__ == '__main__':
  tf.app.run()
