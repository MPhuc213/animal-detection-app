import cv2
from ultralytics import YOLO
import streamlit as st
import numpy as np

# Load model (chỉ 1 lần)
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

model = load_model()

def detect_video(video_path, output_path="output.mp4", conf=0.25, iou=0.45):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Không mở được video!")
        return None, {}

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0: total_frames = 1000  # fallback

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    max_class_count = {}
    frame_count = 0

    progress_bar = st.progress(0)
    status_text = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Detect an toàn 100%
        results = model(frame, conf=conf, iou=iou, verbose=False)[0]
        annotated = results.plot()  # đẹp sẵn

        # Đếm object (an toàn với frame trống)
        current_count = {}
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                class_name = model.names[cls_id]
                current_count[class_name] = current_count.get(class_name, 0) + 1

            # Cập nhật MAX
            for name, cnt in current_count.items():
                max_class_count[name] = max(max_class_count.get(name, 0), cnt)

        out.write(annotated)

        # Cập nhật progress
        progress = frame_count / total_frames
        progress_bar.progress(min(progress, 1.0))
        status_text.text(f"Đang xử lý: {frame_count}/{total_frames} frames ({progress*100:.1f}%)")

    cap.release()
    out.release()
    progress_bar.empty()
    status_text.empty()

    return output_path, max_class_count


def process_video_with_preview(video_path, output_path="output.mp4", show_preview=True, conf=0.25, iou=0.45):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Không mở được video!")
        return None, {}

    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(5) or 30.0
    total_frames = int(cap.get(7)) or 1000

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    max_class_count = {}
    frame_count = 0

    progress_bar = st.progress(0)
    status_text = st.empty()
    preview_placeholder = st.empty() if show_preview else None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        results = model(frame, conf=conf, iou=iou, verbose=False)[0]
        annotated = results.plot()

        # Đếm an toàn
        if results.boxes is not None and len(results.boxes) > 0:
            current_count = {}
            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                name = model.names[cls_id]
                current_count[name] = current_count.get(name, 0) + 1

            for name, cnt in current_count.items():
                max_class_count[name] = max(max_class_count.get(name, 0), cnt)

        out.write(annotated)

        # Progress + preview
        progress = frame_count / total_frames
        progress_bar.progress(min(progress, 1.0))
        status_text.text(f"Đang xử lý: {frame_count}/{total_frames} ({progress*100:.1f}%)")

        if show_preview and frame_count % max(1, total_frames//20) == 0:
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            preview_placeholder.image(annotated_rgb, use_container_width=True)

    cap.release()
    out.release()
    progress_bar.empty()
    status_text.empty()
    if preview_placeholder:
        preview_placeholder.empty()

    return output_path, max_class_count