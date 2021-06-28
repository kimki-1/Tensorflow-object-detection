import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2 
import os 
import pathlib 

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

# 내 로컬에 설치된 TFOD경로 
# PATH_TO_LABELS = 'C:\\Users\\admin\\Documents\\Tensorflow\\models\\research\\object_detection\\data\\mscoco_label_map.pbtxt'
PATH_TO_LABELS = 'C:\\Users\\5-18\\Documents\\Tensorflow\\models\\research\\object_detection\\data\\mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)
# print(category_index)

# 모델이름만 바꾸면 다른 모델을 사용할 수 있는 함수 
def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name, 
        origin=base_url + model_file,
        untar=True)

    model_dir = pathlib.Path(model_dir)/"saved_model"

    model = tf.saved_model.load(str(model_dir))

    return model

## 함수 테스트 (유닛테스트)
model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
detection_model = load_model(model_name)

# print(detection_model.signatures['serving_default'].inputs)
# print(detection_model.signatures['serving_default'].output_dtypes)
# print(detection_model.signatures['serving_default'].output_shapes)

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]
    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                    for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
    # Handle models with masks:
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = tf.convert_to_tensor(output_dict['detection_masks'], dtype=tf.float32)
        output_dict['detection_boxes'] = tf.convert_to_tensor(output_dict['detection_boxes'], dtype=tf.float32)
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                output_dict['detection_masks'], output_dict['detection_boxes'],
                image.shape[0], image.shape[1])  
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                        tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
        
    return output_dict

## 함수 테스트 > 테스트 끝나면 주석처리, 이함수는 다른 부분에서 사용해야하기때문에 
# image_np = cv2.imread('data/images/image1.jpg')
# output_dict = run_inference_for_single_image(detection_model, image_np)
# print(output_dict.keys())

## 예측한 결과를 보여줘라 
def show_inference(model, image_path):

    image_np = cv2.imread(str(image_path))
    # image_np = np.array(Image.open(image_path))
    # 이미지를 오픈으로 받아오면 변경시켜야한다 
    # image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    output_dict = run_inference_for_single_image(model, image_np)
    
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.array(output_dict['detection_boxes']),
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed',None),
        use_normalized_coordinates=True,
        line_thickness=8)
    cv2.imshow(str(image_path), cv2.resize(image_np, (1600, 1000)))
    

## 함수 테스트
PATH_TO_TEST_IMAGE_DIR = pathlib.Path('data\\test')
TEST_IMAGE_PATH = sorted( list(PATH_TO_TEST_IMAGE_DIR.glob("testing*")) )

# show_inference(detection_model, 'data/images/image1.jpg')

for image_path in TEST_IMAGE_PATH:
    show_inference(detection_model, image_path)

cv2.waitKey()
cv2.destroyAllWindows()

