import streamlit as st
from utils.video import process_video_with_preview, detect_video_realtime
import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO

# FORCE RELOAD
import importlib
import sys
if 'utils.video' in sys.modules:
    importlib.reload(sys.modules['utils.video'])
    from utils.video import process_video_with_preview, detect_video_realtime

st.set_page_config(
    page_title="Đếm vật thể - Nhóm 12", 
    layout="wide", 
    page_icon="🎯",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a8a 0%, #3b82f6 100%);
    }
    [data-testid="stSidebar"] .element-container {
        color: white !important;
    }
    .main-header {
        text-align: center;
        color: #1e3a8a;
        padding: 1.5rem 0;
        font-size: 2.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .group-title {
        text-align: center;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        padding: 1rem 0;
        background: rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .stButton>button {
        width: 100%;
    }
    [data-testid="stSidebar"] .stSelectbox label {
        color: white !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# Main title
st.markdown("<h1 class='main-header'>🎯 HỆ THỐNG ĐẾM VẬT THỂ</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    # Logo
    st.markdown("""
        <div style="text-align: center; padding: 20px 0; background-color: #0e1a2f; border-radius: 15px; margin-bottom: 20px;">
            <img src="https://tools1s.com/images/dkmh/vaa-logo.png" width="140">
            <p style="color: white; margin: 15px 0 0 0; font-size: 1.35rem; font-weight: bold; letter-spacing: 1px;">
                Nhóm 12 _ Lập trình Python
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Tiêu đề nhóm
    st.markdown("""
        <div class='group-title'>
            📚 Nhóm 12<br>
            <span style='font-size: 0.9rem;'>ĐẾM vật thể</span>
        </div>
    """, unsafe_allow_html=True)
    
    # CHỌN MODEL
    st.markdown("<p style='color: white; font-weight: bold; font-size: 1.1rem; margin-top: 1rem;'> CHỌN MODEL</p>", unsafe_allow_html=True)
    
    model_folder = "models"
    if os.path.exists(model_folder):
        model_files = glob.glob(os.path.join(model_folder, "*.pt"))
        model_names = [os.path.basename(f) for f in model_files]
        
        if model_names:
            selected_model = st.selectbox(
                "Model:",
                model_names,
                index=model_names.index("best.pt") if "best.pt" in model_names else 0,
                label_visibility="collapsed"
            )
            model_path = os.path.join(model_folder, selected_model)
            
            model_size = os.path.getsize(model_path) / (1024 * 1024)
            st.markdown(f"""
                <div style='background: rgba(255,255,255,0.1); padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0;'>
                    <small style='color: white;'>
                    📦 Kích thước: {model_size:.1f} MB<br>
                    📁 {model_path}
                    </small>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.error("⚠️ Không tìm thấy file model (.pt)")
            model_path = None
    else:
        st.error(f"⚠️ Thư mục '{model_folder}' không tồn tại")
        model_path = None
    
    st.markdown("---")
    
    # Navigation
    st.markdown("<p style='color: white; font-weight: bold; font-size: 1.1rem;'>🧭 CHỨC NĂNG</p>", unsafe_allow_html=True)
    
    option = st.selectbox(
        "Chọn chức năng:",
        ["🖼️ Đếm từ ảnh", "🎥 Đếm từ video", "📈 Visualize Training Results"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Thông tin nhóm
    with st.expander("👥 Thành viên nhóm", expanded=False):
        st.markdown("""
        <div style='color: white;'>
        • Trần Thanh Đạt(Lead)<br>
        • Nguyễn Minh Phúc (Thành Viên)<br>
        • Trần Thanh Trúc (Thành Viên)<br>
        • Đồng Đức Mạnh (Thành Viên)<br>
        • Nguyễn Trần Duy Khánh (Thành Viên)
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("📖 Hướng dẫn", expanded=False):
        st.markdown("""
        <div style='color: white;'>
        <b>🖼️ Đếm từ ảnh:</b><br>
        Upload ảnh để Đếm vật thể<br><br>
        <b>🎥 Đếm từ video:</b><br>
        Upload video để Đếm và đếm vật thể<br><br>
        <b>📈 Visualize:</b><br>
        vật thểm kết quả training model
        </div>
        """, unsafe_allow_html=True)

# -------------------------
# ẢNH
# -------------------------
if option == "🖼️ Đếm từ ảnh":
    st.header("📷 Đếm vật thể từ ảnh")
    
    if model_path is None or not os.path.exists(model_path):
        st.error("❌ Vui lòng chọn model hợp lệ từ sidebar")
        st.stop()
    
    @st.cache_resource
    def load_model(path):
        return YOLO(path)
    
    try:
        model = load_model(model_path)
        st.success(f"✅ Đã load model: {selected_model}")
    except Exception as e:
        st.error(f"❌ Lỗi load model: {str(e)}")
        st.stop()
    
    with st.expander("⚙️ Cài đặt thông số", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            confidence_threshold = st.slider(
                "🎯 Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.05,
                help="Ngưỡng độ tin cậy"
            )
        
        with col2:
            iou_threshold = st.slider(
                "📦 IoU Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.45,
                step=0.05,
                help="Ngưỡng IoU cho NMS"
            )
        
        st.info(f"**Cài đặt:** Confidence ≥ {confidence_threshold:.2f} | IoU ≤ {iou_threshold:.2f}")
    
    upload_files = st.file_uploader(
        "🖼️ Chọn ảnh", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="JPG, JPEG, PNG"
    )
    
    if upload_files:
        for idx, upload in enumerate(upload_files):
            st.markdown(f"### 🖼️ Ảnh {idx + 1}: {upload.name}")
            
            col_left, col_right = st.columns(2)
            
            try:
                file_bytes = upload.read()
                img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
                
                if img is None:
                    st.error(f"❌ Không đọc được ảnh")
                    continue
                
                with col_left:
                    st.markdown("**Ảnh gốc**")
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                with st.spinner("🔍 Đang Đếm..."):
                    results = model(img, conf=confidence_threshold, iou=iou_threshold)[0]
                    annotated = results.plot()
                    
                    class_count = {}
                    for box in results.boxes:
                        cls_id = int(box.cls.item())
                        class_name = model.names[cls_id]
                        class_count[class_name] = class_count.get(class_name, 0) + 1
                
                with col_right:
                    st.markdown("**Kết quả**")
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                if class_count:
                    st.success("✅ Đếm thành công!")
                    with st.expander("📊 Thống kê", expanded=True):
                        cols = st.columns(len(class_count))
                        for idx, (name, count) in enumerate(class_count.items()):
                            with cols[idx]:
                                st.metric(str(name).capitalize(), count)
                        st.bar_chart(class_count)
                else:
                    st.warning("⚠️ Không Đếm được vật thể")
                
                st.markdown("---")
                
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")
    else:
        st.info("👆 Upload ảnh để bắt đầu")

# -------------------------
# VIDEO
# -------------------------
elif option == "🎥 Đếm từ video":
    st.header("🎥 Đếm vật thể từ video")
    
    if model_path is None or not os.path.exists(model_path):
        st.error("❌ Vui lòng chọn model từ sidebar")
        st.stop()
    
    st.success(f"✅ Model: {selected_model}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        upload_files = st.file_uploader(
            "📹 Chọn video", 
            type=["mp4", "avi", "mov"],
            accept_multiple_files=True
        )
    
    with col2:
        st.markdown("**⚙️ Cài đặt:**")
        show_preview = st.checkbox("Preview", value=True)
        save_output = st.checkbox("Lưu video", value=True)
        use_tracking = st.checkbox("Tracking", value=True, help="Đếm unique objects")
    
    with st.expander("🎯 Ngưỡng Đếm", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            confidence_threshold = st.slider("🎯 Confidence", 0.0, 1.0, 0.25, 0.05)
        
        with col2:
            iou_threshold = st.slider("📦 IoU", 0.0, 1.0, 0.45, 0.05)
        
        st.info(f"Conf ≥ {confidence_threshold:.2f} | IoU ≤ {iou_threshold:.2f} | Tracking: {'✅' if use_tracking else '❌'}")
    
    if upload_files:
        for idx, upload in enumerate(upload_files):
            st.markdown(f"### 🎬 Video {idx + 1}: {upload.name}")
            
            try:
                temp_input = f"temp_input_{idx}.mp4"
                with open(temp_input, "wb") as f:
                    f.write(upload.read())
                
                with st.expander("📹 Video gốc", expanded=False):
                    st.video(temp_input)
                
                st.markdown("#### 🔍 Đang xử lý...")
                
                if save_output:
                    output_path = f"output_{idx}_{upload.name}"
                    output_path, class_count = process_video_with_preview(temp_input, output_path, show_preview,conf=confidence_threshold, iou=iou_threshold,model_path=model_path, use_tracking=use_tracking)
                else:
                    class_count = detect_video_realtime(
                        temp_input,
                        conf=confidence_threshold, iou=iou_threshold,
                        model_path=model_path, use_tracking=use_tracking
                    )
                    output_path = None
                
                st.success("✅ Hoàn thành!")
                
                if save_output and output_path and os.path.exists(output_path):
                    st.markdown("#### 🎥 Video đã xử lý")
                    st.video(output_path)
                    
                    with open(output_path, "rb") as file:
                        st.download_button(
                            "⬇️ Tải video",
                            file,
                            f"detected_{upload.name}",
                            "video/mp4",
                            use_container_width=True
                        )
                
                if class_count and isinstance(class_count, dict):
                    with st.expander("📊 Thống kê", expanded=True):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Số lượng {'unique' if use_tracking else 'MAX'}:**")
                            for name, count in sorted(class_count.items(), key=lambda x: x[1], reverse=True):
                                st.metric(str(name).capitalize(), count)
                        
                        with col2:
                            import pandas as pd
                            df = pd.DataFrame(list(class_count.items()), columns=['Class', 'Count'])
                            st.bar_chart(df.set_index('Class'))
                else:
                    st.warning("⚠️ Không đếm được vật thể")
                
                if os.path.exists(temp_input):
                    os.remove(temp_input)
                
                st.markdown("---")
                
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")
                import traceback
                with st.expander("Chi tiết"):
                    st.code(traceback.format_exc())
    else:
        st.info("👆 Upload video")

# -------------------------
# VISUALIZE
# -------------------------
elif option == "📈 Visualize Training Results":
    st.header("📈 Kết quả Training Model")
    
    st.info("""
    📊 **Xem kết quả training YOLO**
    
    Hiển thị các biểu đồ: Confusion Matrix, Curves, Predictions, và nhiều hơn nữa
    """)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        results_path = st.text_input(
            "📁 Đường dẫn thư mục kết quả:",
            value="runs/detect/train",
            help="Ví dụ: runs/detect/train, runs/detect/train2"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        refresh = st.button("🔄 Tải lại", use_container_width=True)
    
    if os.path.exists(results_path):
        st.success(f"✅ Tìm thấy: `{results_path}`")
        
        # Kiểm tra file args.yaml để hiển thị thông tin training
        args_path = os.path.join(results_path, "args.yaml")
        if os.path.exists(args_path):
            with st.expander("ℹ️ Thông tin Training", expanded=False):
                try:
                    import yaml
                    with open(args_path, 'r') as f:
                        args = yaml.safe_load(f)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Epochs", args.get('epochs', 'N/A'))
                        st.metric("Batch Size", args.get('batch', 'N/A'))
                    
                    with col2:
                        st.metric("Image Size", args.get('imgsz', 'N/A'))
                        st.metric("Model", args.get('model', 'N/A'))
                    
                    with col3:
                        st.metric("Optimizer", args.get('optimizer', 'N/A'))
                        st.metric("LR0", args.get('lr0', 'N/A'))
                    
                    with col4:
                        st.metric("Workers", args.get('workers', 'N/A'))
                        st.metric("Device", args.get('device', 'N/A'))
                    
                except Exception as e:
                    st.warning(f"⚠️ Không đọc được args.yaml: {e}")
        
        # Tabs hiển thị kết quả
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Confusion Matrix", 
            "📈 Training Curves", 
            "🎯 Predictions", 
            "📉 Performance Metrics",
            "📂 All Files"
        ])
        
        # TAB 1: Confusion Matrix
        with tab1:
            st.subheader("🔢 Ma trận nhầm lẫn (Confusion Matrix)")
            
            st.markdown("""
            Ma trận nhầm lẫn cho thấy mô hình phân loại đúng/sai như thế nào cho từng class.
            - **Hàng**: Ground truth (nhãn thực tế)
            - **Cột**: Predictions (dự đoán của model)
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                cm_path = os.path.join(results_path, "confusion_matrix.png")
                if os.path.exists(cm_path):
                    st.image(cm_path, caption="Confusion Matrix", use_container_width=True)
                    
                    with open(cm_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download Confusion Matrix",
                            f,
                            "confusion_matrix.png",
                            "image/png",
                            use_container_width=True
                        )
                else:
                    st.warning("⚠️ Không tìm thấy confusion_matrix.png")
            
            with col2:
                cm_norm_path = os.path.join(results_path, "confusion_matrix_normalized.png")
                if os.path.exists(cm_norm_path):
                    st.image(cm_norm_path, caption="Normalized Confusion Matrix", use_container_width=True)
                    
                    with open(cm_norm_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download Normalized Matrix",
                            f,
                            "confusion_matrix_normalized.png",
                            "image/png",
                            use_container_width=True
                        )
                else:
                    st.warning("⚠️ Không tìm thấy confusion_matrix_normalized.png")
            
            st.info("""
            💡 **Cách đọc:**
            - Đường chéo chính (từ trái trên xuống phải dưới): Dự đoán đúng
            - Các ô ngoài đường chéo: Dự đoán sai (confusion)
            - Normalized matrix hiển thị tỷ lệ % thay vì số lượng
            """)
        
        # TAB 2: Training Curves
        with tab2:
            st.subheader("📉 Đường cong Training")
            
            results_img = os.path.join(results_path, "results.png")
            if os.path.exists(results_img):
                st.image(results_img, caption="Training & Validation Metrics", use_container_width=True)
                
                st.markdown("""
                **Các metrics quan trọng:**
                - **Box Loss**: Độ chính xác vị trí bounding box
                - **Class Loss**: Độ chính xác phân loại
                - **mAP50**: Mean Average Precision @ IoU 0.5
                - **mAP50-95**: mAP trung bình từ IoU 0.5 đến 0.95
                """)
                
                with open(results_img, "rb") as f:
                    st.download_button(
                        "⬇️ Download Results Chart",
                        f,
                        "training_results.png",
                        "image/png",
                        use_container_width=True
                    )
            else:
                st.warning("⚠️ Không tìm thấy results.png")
            
            st.markdown("---")
            st.subheader("🎯 Precision & Recall Curves")
            
            col1, col2 = st.columns(2)
            
            with col1:
                pr_path = os.path.join(results_path, "PR_curve.png")
                if os.path.exists(pr_path):
                    st.image(pr_path, caption="Precision-Recall Curve", use_container_width=True)
                    st.caption("**PR Curve**: Quan hệ giữa Precision và Recall")
                else:
                    st.info("ℹ️ Không có PR_curve.png")
                
                p_path = os.path.join(results_path, "P_curve.png")
                if os.path.exists(p_path):
                    st.image(p_path, caption="Precision Curve", use_container_width=True)
                else:
                    st.info("ℹ️ Không có P_curve.png")
            
            with col2:
                f1_path = os.path.join(results_path, "F1_curve.png")
                if os.path.exists(f1_path):
                    st.image(f1_path, caption="F1 Score Curve", use_container_width=True)
                    st.caption("**F1 Score**: Trung bình điều hòa của Precision và Recall")
                else:
                    st.info("ℹ️ Không có F1_curve.png")
                
                r_path = os.path.join(results_path, "R_curve.png")
                if os.path.exists(r_path):
                    st.image(r_path, caption="Recall Curve", use_container_width=True)
                else:
                    st.info("ℹ️ Không có R_curve.png")
        
        # TAB 3: Predictions
        with tab3:
            st.subheader("🎯 Ví dụ Predictions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📋 Labels Distribution")
                labels_path = os.path.join(results_path, "labels.jpg")
                if os.path.exists(labels_path):
                    st.image(labels_path, caption="Phân bố nhãn trong dataset", use_container_width=True)
                else:
                    st.info("ℹ️ Không có labels.jpg")
                
                st.markdown("#### 🎓 Training Batch")
                train_batch = os.path.join(results_path, "train_batch0.jpg")
                if os.path.exists(train_batch):
                    st.image(train_batch, caption="Ảnh training batch đầu tiên", use_container_width=True)
                else:
                    st.info("ℹ️ Không có train_batch0.jpg")
            
            with col2:
                st.markdown("#### ✅ Validation Labels")
                val_labels = os.path.join(results_path, "val_batch0_labels.jpg")
                if os.path.exists(val_labels):
                    st.image(val_labels, caption="Ground truth labels", use_container_width=True)
                else:
                    st.info("ℹ️ Không có val_batch0_labels.jpg")
                
                st.markdown("#### 🔮 Validation Predictions")
                val_pred = os.path.join(results_path, "val_batch0_pred.jpg")
                if os.path.exists(val_pred):
                    st.image(val_pred, caption="Model predictions", use_container_width=True)
                else:
                    st.info("ℹ️ Không có val_batch0_pred.jpg")
            
            # Hiển thị thêm các validation batch khác
            st.markdown("---")
            st.markdown("#### 📸 Các validation batch khác")
            
            val_batches = glob.glob(os.path.join(results_path, "val_batch*_pred.jpg"))
            if len(val_batches) > 1:
                cols = st.columns(3)
                for idx, batch_path in enumerate(val_batches[1:6]):  # Hiển thị 5 batch tiếp theo
                    with cols[idx % 3]:
                        st.image(batch_path, caption=os.path.basename(batch_path), use_container_width=True)
            else:
                st.info("ℹ️ Chỉ có 1 validation batch")
        
        # TAB 4: Performance Metrics
        with tab4:
            st.subheader("📊 Các metrics hiệu suất")
            
            # Đọc file results.csv nếu có
            csv_path = os.path.join(results_path, "results.csv")
            if os.path.exists(csv_path):
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_path)
                    
                    # Lấy metrics từ epoch cuối
                    last_epoch = df.iloc[-1]
                    
                    st.markdown("#### 🏆 Kết quả epoch cuối cùng")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if 'metrics/mAP50(B)' in df.columns:
                            st.metric("mAP@0.5", f"{last_epoch['metrics/mAP50(B)']:.3f}")
                        if 'metrics/precision(B)' in df.columns:
                            st.metric("Precision", f"{last_epoch['metrics/precision(B)']:.3f}")
                    
                    with col2:
                        if 'metrics/mAP50-95(B)' in df.columns:
                            st.metric("mAP@0.5:0.95", f"{last_epoch['metrics/mAP50-95(B)']:.3f}")
                        if 'metrics/recall(B)' in df.columns:
                            st.metric("Recall", f"{last_epoch['metrics/recall(B)']:.3f}")
                    
                    with col3:
                        if 'train/box_loss' in df.columns:
                            st.metric("Box Loss", f"{last_epoch['train/box_loss']:.4f}")
                        if 'train/cls_loss' in df.columns:
                            st.metric("Class Loss", f"{last_epoch['train/cls_loss']:.4f}")
                    
                    with col4:
                        if 'val/box_loss' in df.columns:
                            st.metric("Val Box Loss", f"{last_epoch['val/box_loss']:.4f}")
                        if 'val/cls_loss' in df.columns:
                            st.metric("Val Class Loss", f"{last_epoch['val/cls_loss']:.4f}")
                    
                    st.markdown("---")
                    st.markdown("#### 📈 Lịch sử Training")
                    
                    # Hiển thị bảng
                    with st.expander("📋 Xem bảng chi tiết", expanded=False):
                        st.dataframe(df, use_container_width=True)
                    
                    # Download CSV
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Download Results CSV",
                        csv_data,
                        "training_results.csv",
                        "text/csv",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.warning(f"⚠️ Không đọc được results.csv: {e}")
            else:
                st.info("ℹ️ Không tìm thấy results.csv")
            
            # Hiển thị labels correlogram
            st.markdown("---")
            st.markdown("#### 🔗 Labels Correlogram")
            
            correlogram_path = os.path.join(results_path, "labels_correlogram.jpg")
            if os.path.exists(correlogram_path):
                st.image(correlogram_path, caption="Mối tương quan giữa các class", use_container_width=True)
                st.caption("Cho biết các class nào thường xuất hiện cùng nhau")
            else:
                st.info("ℹ️ Không có labels_correlogram.jpg")
        
        # TAB 5: All Files
        with tab5:
            st.subheader("📂 Tất cả file trong thư mục")
            
            # Liệt kê tất cả file
            all_files = []
            for root, dirs, files in os.walk(results_path):
                for file in files:
                    all_files.append(os.path.join(root, file))
            
            if all_files:
                # Phân loại file
                images = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                csvs = [f for f in all_files if f.lower().endswith('.csv')]
                yamls = [f for f in all_files if f.lower().endswith('.yaml')]
                pts = [f for f in all_files if f.lower().endswith('.pt')]
                others = [f for f in all_files if f not in images + csvs + yamls + pts]
                
                # Metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("🖼️ Images", len(images))
                with col2:
                    st.metric("📊 CSV", len(csvs))
                with col3:
                    st.metric("⚙️ YAML", len(yamls))
                with col4:
                    st.metric("🤖 Model", len(pts))
                with col5:
                    st.metric("📦 Khác", len(others))
                
                # Hiển thị danh sách file
                with st.expander("📋 Danh sách chi tiết", expanded=False):
                    import pandas as pd
                    
                    file_data = []
                    for f in all_files:
                        file_size = os.path.getsize(f)
                        size_mb = file_size / (1024 * 1024)
                        
                        file_data.append({
                            'Tên': os.path.basename(f),
                            'Đường dẫn': f,
                            'Kích thước': f"{size_mb:.2f} MB" if size_mb >= 1 else f"{file_size/1024:.1f} KB",
                            'Loại': os.path.splitext(f)[1]
                        })
                    
                    df = pd.DataFrame(file_data)
                    st.dataframe(df, use_container_width=True, height=400)
                
                # Hiển thị tất cả ảnh
                if images:
                    st.markdown("---")
                    st.markdown("#### 🖼️ Tất cả ảnh trong thư mục")
                    
                    # Tùy chọn số cột
                    num_cols = st.slider("Số cột hiển thị:", 2, 5, 3)
                    
                    cols = st.columns(num_cols)
                    for idx, img_path in enumerate(sorted(images)):
                        with cols[idx % num_cols]:
                            st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
                            
                            # Hiển thị size
                            file_size = os.path.getsize(img_path) / 1024
                            st.caption(f"📦 {file_size:.1f} KB")
                
                # Download all as ZIP
                st.markdown("---")
                st.markdown("### 📥 Download toàn bộ")
                
                if st.button("📦 Tạo file ZIP", use_container_width=True):
                    import zipfile
                    import io
                    
                    with st.spinner("Đang nén file..."):
                        zip_buffer = io.BytesIO()
                        
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            for file_path in all_files:
                                arcname = os.path.relpath(file_path, results_path)
                                zip_file.write(file_path, arcname)
                        
                        st.download_button(
                            "⬇️ Download ZIP",
                            zip_buffer.getvalue(),
                            f"{os.path.basename(results_path)}_results.zip",
                            "application/zip",
                            use_container_width=True
                        )
                        
                        st.success(f"✅ Đã tạo file ZIP với {len(all_files)} files")
            else:
                st.warning("⚠️ Thư mục trống")
                
    else:
        st.error(f"❌ Không tìm thấy thư mục: `{results_path}`")
        
        st.markdown("---")
        st.markdown("### 💡 Hướng dẫn:")
        st.markdown("""
        Sau khi training YOLO, kết quả thường được lưu tại:
        
        ```
        runs/detect/train/          # Lần train đầu tiên
        runs/detect/train2/         # Lần train thứ 2
        runs/detect/train3/         # Lần train thứ 3
        ...
        ```
        
        **Cấu trúc thư mục kết quả:**
        ```
        runs/detect/train/
        ├── weights/
        │   ├── best.pt           # Model tốt nhất
        │   └── last.pt           # Model epoch cuối
        ├── confusion_matrix.png
        ├── results.png
        ├── PR_curve.png
        ├── F1_curve.png
        ├── results.csv
        ├── args.yaml
        └── [các file khác...]
        ```
        
        📝 **Nhập đường dẫn chính xác vào ô trên để xem kết quả!**
        """)





# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🎯 Obj detection - Nhóm 12</p>
</div>
""", unsafe_allow_html=True)