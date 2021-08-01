import cv2
from detect import create_names_dictionary,configure,do_image_processing,test_drawing,do_video_processing



def test_image_detection():
    img_path="data/humans.jpg"
    frame_test =cv2.imread(img_path)
    labels=create_names_dictionary("./data/coco.names")
    
    
    saved_model_loaded,infer,configuration= configure()


    detection_results= do_image_processing(frame_test,infer,configuration,labels)

    test_drawing(img_path,detection_results)

    print(detection_results)


    

def test_video_processing():
    
    labels=create_names_dictionary("./data/coco.names")
    
    
    saved_model_loaded,infer,configuration= configure()

    do_video_processing(0,infer,configuration,labels)

    


#call any of the two functions to see the results
test_video_processing()