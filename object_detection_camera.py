import tensorflow as tf 
print(tf.__version__)

import tarfile 
import urllib.request 
import os 

# 모델 다운로드 하고 압축 푸는 코드 
# MODEL_DATE = '20200711'
# MODEL_NAME = 'mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8'
# MODEL_NAME = 'faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8'
# MODEL_NAME = 'ssd_resnet152_v1_fpn_640x640_coco17_tpu-8'

# MODEL_NAME = 'ssd_mobilenet_v2_320x320_coco17_tpu-8'
MODEL_TAR_FILENAME = MODEL_NAME + '.tar.gz'
MODELS_DIR = 'data/models'

# tf zoo, 모델 다운로드 받을 수 있는곳 
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

# http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz

MODELS_DOMWLOAD_BASE = 'http://download.tensorflow.org/models/object_detection'
# MODEL_DOWNLOAD_LINK = MODELS_DOMWLOAD_BASE + MODEL_DATE + '/' + MODEL_TAR_FILENAME
MODEL_DOWNLOAD_LINK = MODELS_DOMWLOAD_BASE + '/' + MODEL_TAR_FILENAME

PATH_TO_MODEL_TAR = os.path.join('data/models', MODEL_TAR_FILENAME)
PATH_TO_CKPT = os.path.join('data/models', os.path.join(MODEL_NAME, 'checkpoint/'))
PATH_TO_CFG = os.path.join('data/models', os.path.join(MODEL_NAME, 'pipeline.config'))

# 모델받아서 압축풀기 
if not os.path.exists(PATH_TO_CKPT):
    print('Downloading model. This may take a while... ', end='')
    urllib.request.urlretrieve(MODEL_DOWNLOAD_LINK, PATH_TO_MODEL_TAR)
    tar_file = tarfile.open(PATH_TO_MODEL_TAR)
    tar_file.extractall(MODELS_DIR)
    tar_file.close()
    os.remove(PATH_TO_MODEL_TAR)
    print('Done')

# 레이블 다운로드 받기 
LABEL_FILENAME = 'mscoco_label_map.pbtxt'
LABELS_DOWNLOAD_BASE = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
PATH_TO_LABELS = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, LABEL_FILENAME))

if not os.path.exists(PATH_TO_LABELS):
    print('Downloading label file... ', end='')
    urllib.request.urlretrieve(LABELS_DOWNLOAD_BASE + LABEL_FILENAME, PATH_TO_LABELS)
    print('Done')

## 모델 로딩
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

# GPU의 Dynamic Memory Allocation을 활성화 시키는 코드
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus :
    tf.config.experimental.set_memory_growth(gpu, True)

# config 로드하고, 모델빌드
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore Checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore( os.path.join(PATH_TO_CKPT, 'ckpt-0') ).expect_partial()

@tf.function
def detect_fn(image) :
    """Detect Objects in Image."""
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])


## 레이블 맵 데이터 로딩 ( Load label map data )
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)

# print(category_index)

# 자율주행자동차 장치에 이코드 쓰면된다
import cv2 
import numpy as np 
import pandas as pd 

cap = cv2.VideoCapture('data/videos/good2.mp4')

while True :
    ret, image_np = cap.read()
    # (1, x, x, 3)
    image_np_expended = np.expand_dims(image_np, axis=0)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    # mscoco_label_map.pdtxt파일을 보면, id가 1부터 시작하니까 
    # offset을 1로 만들어 준다. 
    label_id_offset = 1 

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates = True,
        max_boxes_to_draw = 200,
        # min_score_threshold=0.6
    )

    # cv2.imshow('object detection', image_np_with_detections)
    
    # 관제 시스템의 경우 디스플레이 크면 거기에 맞게 조절해서 보여주도록 하는 코드 
    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

    if cv2.waitKey(25) & 0xFF == 27 :
        break

cap.release()
cv2.destroyAllWindows()

