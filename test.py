from detect import create_names_dictionary,configure,do_image_processing,test_drawing,do_video_processing

humans_picture=[{'class': 'person', 'id': 0, 'confidence': 0.98476183, 'x': 352, 'y': 149, 'height': 472, 'width': 445}, {'class': 'car', 'id': 2, 'confidence': 0.9130749, 'x': 3, 'y': 437, 'height': 659, 'width': 163}, {'class': 'person', 'id': 0, 'confidence': 0.88458365, 'x': 493, 'y': 204, 'height': 462, 'width': 570}, {'class': 'person', 'id': 0, 'confidence': 0.85075754, 'x': 645, 'y': 200, 'height': 466, 'width': 728}, {'class': 'traffic light', 'id': 9, 'confidence': 0.68847036, 'x': 599, 'y': 0, 'height': 66, 'width': 715}, {'class': 'car', 'id': 2, 'confidence': 0.6133746, 'x': 475, 'y': 0, 'height': 29, 'width': 610}]

def test_image_detection():
    img_path="data/humans.jpg"
    frame_test =cv2.imread(img_path)
    labels=create_names_dictionary("./data/coco.names")
    
    
    saved_model_loaded,infer,configuration= configure()


    detection_results= do_image_processing(frame_test,infer,configuration,labels)

    test_drawing(img_path,detection_results)

    assert detection_results==humans_picture

def test_video_processing():
    
    labels=create_names_dictionary("./data/coco.names")
    
    
    saved_model_loaded,infer,configuration= configure()

    do_video_processing(0,infer,configuration,labels)
