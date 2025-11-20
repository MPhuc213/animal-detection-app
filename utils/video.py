import cv2
from ultralytics import YOLO

model = YOLO("models/best.pt")

def detect_video(video_path, output_path="output.mp4"):
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Kiểm tra FPS hợp lệ
    if fps <= 0 or fps > 120:
        fps = 30.0  # Mặc định 30 FPS nếu không đọc được

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    class_count = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        annotated = results.plot()

        # Đếm class theo TÊN thay vì ID
        for box in results.boxes:
            cls_id = int(box.cls.item())
            # Lấy tên class từ model
            class_name = model.names[cls_id]
            class_count[class_name] = class_count.get(class_name, 0) + 1

        out.write(annotated)

    cap.release()
    out.release()
    
    return output_path, class_count