from ultralytics import YOLO
import cv2

model = YOLO("models/best.pt")

def detect_image(image):
    """
    Phát hiện động vật trong ảnh
    
    Args:
        image: numpy array của ảnh (BGR format)
    
    Returns:
        annotated: ảnh đã được vẽ bounding box
        class_count: dictionary đếm số lượng từng loại động vật
    """
    results = model(image)[0]
    annotated = results.plot()
    
    class_count = {}
    
    # Đếm số lượng theo TÊN class
    for box in results.boxes:
        cls_id = int(box.cls.item())
        # Lấy tên class từ model
        class_name = model.names[cls_id]
        class_count[class_name] = class_count.get(class_name, 0) + 1
    
    return annotated, class_count