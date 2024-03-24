from ultralytics import YOLO

# model = YOLO('yolov8x.pt') #creating an instance of the yolo model

# model = YOLO('models/yolo5_last.pt') #creating an instance of a trained yolo model, we got the weights from the training process and we're using the weights by specifying the path to the weights file

# result = model.predict('input/input_video.mp4', save = True) #predicting the image/video

# result = model.predict('input/input_video.mp4', conf = 0.2, save = True) #predicting the image/video with a confidence threshold of 0.2

model = YOLO('yolov8x.pt') #creating an instance of the yolo model

result = model.track('input/input_video.mp4', conf = 0.2, save = True) #track the image/video with a confidence threshold of 0.2, tracking 
# is used to track the objects in the video, which means the bounding box of an object in a frame is connected to the bounding box of the same object in the next frame

# print(result) 
# print('bounding boxes:')
# for box in result[0].boxes:
#     print(box)
