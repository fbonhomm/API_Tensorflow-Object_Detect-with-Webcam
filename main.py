import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import sys
import tensorflow as tf
import cv2

sys.path.insert(0, r'object_detection')

from utils import label_map_util
from utils import visualization_utils as vis_util

with tf.device("/gpu:0"):
    MODEL_NAME = 'object_detection/ssd_mobilenet_v1_coco_11_06_2017'
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = os.path.join('object_detection/data', 'mscoco_label_map.pbtxt')
    NUM_CLASSES = 90


    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    def load_image_into_numpy_array(image):
        (im_width, im_height, dim) = image.shape
        return np.array(image).reshape((im_height, im_width, 3)).astype(np.uint8)

    camera = cv2.VideoCapture(0)
    camera.set(cv2.cv.CV_CAP_PROP_FOURCC, cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'))

    print ' ---------- push Q for quit ----------'

    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
          while True:
              ret, image = camera.read()

              image = cv2.resize(image, dsize=(800, 800), interpolation=cv2.INTER_LINEAR)

              image_np = load_image_into_numpy_array(image)
              image_np_expanded = np.expand_dims(image_np, axis=0)
              image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
              boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
              scores = detection_graph.get_tensor_by_name('detection_scores:0')
              classes = detection_graph.get_tensor_by_name('detection_classes:0')
              num_detections = detection_graph.get_tensor_by_name('num_detections:0')

              (boxes, scores, classes, num_detections) = sess.run(
                  [boxes, scores, classes, num_detections],
                  feed_dict={image_tensor: image_np_expanded})

              vis_util.visualize_boxes_and_labels_on_image_array(
                  image_np,
                  np.squeeze(boxes),
                  np.squeeze(classes).astype(np.int32),
                  np.squeeze(scores),
                  category_index,
                  use_normalized_coordinates=True,
                  line_thickness=8)
              cv2.imshow('image', image_np)
              if cv2.waitKey(1) & 0xFF == ord('q'):
                  break
