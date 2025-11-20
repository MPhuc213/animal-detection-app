from ultralytics import YOLO
import cv2

model = YOLO("models/best.pt")

def detect_image(image, conf=0.25, iou=0.45):
    """
    Phát hiện vật thể trong ảnh với confidence và IoU threshold
    
    Args:
        image: numpy array của ảnh (BGR format)
        conf: Confidence threshold (0-1), mặc định 0.25
        iou: IoU threshold cho NMS (0-1), mặc định 0.45
    
    Returns:
        annotated: ảnh đã được vẽ bounding box
        class_count: dictionary đếm số lượng từng loại vật thể
    """
    # Chạy model với confidence và iou threshold
    results = model(image, conf=conf, iou=iou)[0]
    annotated = results.plot()
    
    class_count = {}
    
    # Đếm số lượng object trong ảnh
    for box in results.boxes:
        cls_id = int(box.cls.item())
        class_name = model.names[cls_id]
        class_count[class_name] = class_count.get(class_name, 0) + 1
    
    return annotated, class_count