from ultralytics import YOLO

model = YOLO('yolov8x.pt') #creating an instance of the yolo model

result = model.predict('input/input_video.mp4', save = True) #predicting the image/video

print(result) 
print('bounding boxes:')
for box in result[0].boxes:
    print(box)
