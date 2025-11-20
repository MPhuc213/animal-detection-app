import cv2
from ultralytics import YOLO
import streamlit as st
import tempfile
import os

model = YOLO("models/best.pt")

def detect_video(video_path, output_path="output.mp4"):
    """
    Phát hiện vật thể trong video với progress bar
    """
    cap = cv2.VideoCapture(video_path)
    
    # Đọc thông tin video
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Kiểm tra FPS hợp lệ
    if fps <= 0 or fps > 120:
        fps = 30.0  # Mặc định 30 FPS nếu không đọc được
    
    # Tạo video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    class_count = {}
    frame_count = 0
    
    # Tạo progress bar trong Streamlit
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect
        results = model(frame)[0]
        annotated = results.plot()
        
        # Đếm class theo TÊN
        for box in results.boxes:
            cls_id = int(box.cls.item())
            class_name = model.names[cls_id]
            class_count[class_name] = class_count.get(class_name, 0) + 1
        
        # Ghi frame
        out.write(annotated)
        
        # Update progress
        if total_frames > 0:
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Đã xử lý: {frame_count}/{total_frames} frames ({progress*100:.1f}%)")
    
    cap.release()
    out.release()
    
    # Xóa progress bar
    progress_bar.empty()
    status_text.empty()
    
    return output_path, class_count


def detect_video_realtime(video_path):
    """
    Phát hiện video theo thời gian thực với hiển thị frame by frame
    (Chỉ dùng cho video ngắn hoặc demo)
    """
    cap = cv2.VideoCapture(video_path)
    
    class_count = {}
    frame_count = 0
    
    # Container để hiển thị video
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect
        results = model(frame)[0]
        annotated = results.plot()
        
        # Đếm class
        for box in results.boxes:
            cls_id = int(box.cls.item())
            class_name = model.names[cls_id]
            class_count[class_name] = class_count.get(class_name, 0) + 1
        
        # Hiển thị frame (chuyển BGR -> RGB)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)
        
        # Hiển thị stats
        stats_text = f"Frame: {frame_count}/{total_frames}"
        if class_count:
            stats_text += " | Phát hiện: " + ", ".join([f"{k}: {v}" for k, v in class_count.items()])
        stats_placeholder.text(stats_text)
        
        # Giới hạn tốc độ hiển thị
        if frame_count % 3 == 0:  # Chỉ hiển thị mỗi 3 frame
            continue
    
    cap.release()
    
    return class_count


def process_video_with_preview(video_path, output_path="output.mp4", show_preview=True):
    """
    Xử lý video với tùy chọn preview hoặc chỉ lưu
    """
    cap = cv2.VideoCapture(video_path)
    
    # Đọc thông tin video
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0 or fps > 120:
        fps = 30.0
    
    # Tạo video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    class_count = {}
    frame_count = 0
    
    # UI elements
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if show_preview:
        preview_placeholder = st.empty()
        preview_interval = max(1, total_frames // 20)  # Hiển thị 20 frame mẫu
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect
        results = model(frame)[0]
        annotated = results.plot()
        
        # Đếm class
        for box in results.boxes:
            cls_id = int(box.cls.item())
            class_name = model.names[cls_id]
            class_count[class_name] = class_count.get(class_name, 0) + 1
        
        # Ghi frame
        out.write(annotated)
        
        # Update progress
        if total_frames > 0:
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"⏳ Đang xử lý: {frame_count}/{total_frames} frames ({progress*100:.1f}%)")
            
            # Preview một số frame
            if show_preview and frame_count % preview_interval == 0:
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                preview_placeholder.image(annotated_rgb, caption=f"Frame {frame_count}", use_container_width=True)
    
    cap.release()
    out.release()
    
    # Cleanup UI
    progress_bar.empty()
    status_text.empty()
    if show_preview:
        preview_placeholder.empty()
    
    return output_path, class_count