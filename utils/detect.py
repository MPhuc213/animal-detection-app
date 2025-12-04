from ultralytics import YOLO
import cv2

model = YOLO("model_path")

def detect_image(image, conf=0.25, iou=0.45):
    results = model(image, conf=conf, iou=iou, verbose=False)[0]
    annotated = results.plot()
    class_count = {}
    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls.item())
            class_name = model.names[cls_id]
            class_count[class_name] = class_count.get(class_name, 0) + 1

    return annotated, class_count