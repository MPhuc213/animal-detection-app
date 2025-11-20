from ultralytics import YOLO
import cv2

model = YOLO("models/best.pt")

def detect_image(image):
    """
    Phát hiện vật thể trong ảnh
    """
    results = model(image)[0]
    annotated = results.plot()
    
    class_count = {}
    
    # Đếm số lượng object TRONG ẢNH NÀY (không phải tổng)
    for box in results.boxes:
        cls_id = int(box.cls.item())
        class_name = model.names[cls_id]
        class_count[class_name] = class_count.get(class_name, 0) + 1
    
    return annotated, class_count