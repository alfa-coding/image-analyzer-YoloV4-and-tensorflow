from os import name
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

from tensorflow.python.saved_model import tag_constants
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def format_boxes(bboxes, image_height, image_width):
    for box in bboxes:
        ymin = int(box[0] * image_height)
        xmin = int(box[1] * image_width)
        ymax = int(box[2] * image_height)
        xmax = int(box[3] * image_width)
        width = xmax - xmin
        height = ymax - ymin
        box[0], box[1], box[2], box[3] = xmin, ymin, width, height
    return bboxes

def configure():

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    configuration={

    'input_size' : 416,
    'weights': "./checkpoints/yolov4-tiny-416",
    'model':"yolov4",
    'tiny' :True,
    'iou': 0.45,
    'score':0.50,
    }

   
    # load standard tensorflow saved model
    saved_model_loaded = tf.saved_model.load(configuration['weights'], tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    return saved_model_loaded,infer,configuration


def create_names_dictionary(path:str)->list:
    labels= []
    with open(path,"r") as file:
        for line in file:
            labels.append(line.strip())
    
    return labels

def buildObjectDetected(class_id,class_name, confidence, x, y, x_plus_w, y_plus_h):
    label = class_name
    obj={    }
    obj["class"]=label
    obj['id']=class_id
    obj["confidence"]=confidence
    obj["x"]=x
    obj["y"]=y
    obj["height"]=y_plus_h
    obj["width"]=x_plus_w


    return obj

def do_image_processing(frame,infer,configuration,labels):

    input_size= configuration['input_size']
    iou=configuration['iou']
    score=configuration['score']


    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)

    frame_size = frame.shape[:2]
    image_data = cv2.resize(frame, (input_size, input_size))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    
    
    batch_data = tf.constant(image_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=iou,
        score_threshold=score
    )
    
    # convert data to numpy arrays and slice out unused elements
    num_objects = valid_detections.numpy()[0]
    bboxes = boxes.numpy()[0]
    bboxes = bboxes[0:int(num_objects)]
    scores = scores.numpy()[0]
    scores = scores[0:int(num_objects)]
    classes = classes.numpy()[0]
    classes = classes[0:int(num_objects)]

    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
    original_h, original_w, _ = frame.shape
    bboxes = format_boxes(bboxes, original_h, original_w)
    
    # store all predictions in one parameter for simplicity when calling functions
    pred_bbox = [bboxes, scores, classes, num_objects]
    
   
   
    response=[]
    
    for index,id in enumerate(classes):
        class_name=labels[int(id)]
        bbox=bboxes[index]
        col=int(bbox[0])
        row=int(bbox[1])
        width=col+int(bbox[2])
        heigth=row+ int(bbox[3])
        newObj=buildObjectDetected(int(id),class_name, scores[index], col, row, width, heigth)
        response.append(newObj)
        
    
    return response

def draw_image_anotations(response_with_annotations,img):
    for annotation_info in response_with_annotations:

        label=annotation_info["class"]
        class_id=annotation_info['id']
        confidence=annotation_info["confidence"]
        x=annotation_info["x"]
        y=annotation_info["y"]
        y_plus_h=annotation_info["height"]
        x_plus_w=annotation_info["width"]


       

        color = (255,0,255)

        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img

def test_drawing(img_path,annotations):
    img=cv2.imread(img_path)
    if img.size ==0:
        print("Failed to load frame")
        return
    else:
        annotated_img=draw_image_anotations(annotations,img)
        cv2.imshow("annotations",annotated_img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            print("Exiting")

def do_video_processing(webcam,infer,configuration,labels):
    cap= cv2.VideoCapture(webcam)
    while True:        
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret==False:
            break
        else:
            #process the resulting frame
            detection_results= do_image_processing(frame,infer,configuration,labels)
            annotated_img=draw_image_anotations(detection_results,frame)
            cv2.imshow("annotations",annotated_img)

        #allowing the programmer to quit the process by pressing q
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows() 

