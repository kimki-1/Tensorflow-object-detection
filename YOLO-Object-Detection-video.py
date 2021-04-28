import time 
import cv2
import numpy as np 

from yolo.model.yolo_model import YOLO

def process_image(img) :
    """ 이미지 리사이즈 하고, 차원확장 
    img : 원본 이미지
    결과는 (64, 64, 3)으로 프로세싱된 이미지 변환 """
    
    image_org = cv2.resize(img, (416, 416), interpolation=cv2.INTER_CUBIC)
    image_org = np.array(image_org, dtype='float32')
    image_org = image_org / 255.0
    image_org = np.expand_dims(image_org, axis=0)

    return image_org 

def get_classes(file) :
    """ 클래스의 이름을 가져온다. 
    리스트로 클래스 이름을 반환한다. """

    with open(file) as f :
        name_of_class = f.readlines()
    
    name_of_class = [ class_name.strip() for class_name in name_of_class ]

    return name_of_class

def box_draw(image, boxes, scores, classes, all_classes):
    """ image : 오리지날 이미지 
        boxes : 오브젝트의 박스데이터, ndarray
        classes : 오브젝트의 클래스 정보, ndarray
        scores : 오브젝트의 확률, ndarray 
        all_classes : 모든 클래스 이름 """

    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 255), 1,
                    cv2.LINE_AA)

        

def detect_image(image, yolo, all_classes) : 
    """ image : 오리지날 이미지
        yolo : 욜로 모델 
        all_classes : 전체 클래스 이름 

        변환된 이미지 리턴! """

    pimage = process_image(image)

    image_boxes, image_classes, image_scores = yolo.predict(pimage, image.shape)

    if image_boxes is not None : 
        box_draw(image, image_boxes, image_scores, image_classes, all_classes)
    
    return image 

cap = cv2.VideoCapture("data/videos/test11.mp4")

## 카메라의 영상을 실행하는 코드
# cap = cv2.VideoCapture(0)

if cap.isOpened() == False :
    print("Error opening video stream of file") 

else :
    # 프레임의 정보 가져오기 : 화면 크기 ( width, height )
    frame_width = int (cap.get(3)) 
    frame_height = int(cap.get(4))

    # 캠으로 들어온 비디오를 저장하는 코드 
    out = cv2.VideoWriter("data/videos/YOLO3.mp4", 
                    cv2.VideoWriter_fourcc(*'H264'), 
                    10,
                    (frame_width, frame_height) )

    while cap.isOpened() :

        ret, frame = cap.read()

        if ret == True : 
            strat_time = time.time()
            ### 이부분을 모델 추론하고 화면에 보여주는 코드로 변경 

            yolo = YOLO(0.6, 0.5)
            all_classes = get_classes('yolo/data/coco_classes.txt')
            result_image = detect_image(frame, yolo, all_classes)
            cv2.imshow('result', result_image)
            out.write(result_image)
            end_time = time.time()
            print(end_time - strat_time)
            
            if cv2.waitKey(25) & 0xFF==27 :
                break

        else :
            break

    cap.release()
    cv2.destroyAllWindows() 

