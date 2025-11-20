import streamlit as st
from utils.detect import detect_image
from utils.video import detect_video
from utils.analysis import *
import os

st.set_page_config(
    page_title="Phát hiện vật thể - Nhóm 12", 
    layout="wide", 
    page_icon="🎯",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
    <style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a8a 0%, #3b82f6 100%);
    }
    
    [data-testid="stSidebar"] .element-container {
        color: white !important;
    }
    
    /* Logo container */
    .logo-container {
        text-align: center;
        padding: 1.5rem 0;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    
    /* Title styling */
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
    
    /* Nav items */
    .nav-item {
        background: rgba(255, 255, 255, 0.1);
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .nav-item:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateX(5px);
    }
    
    /* Main header */
    .main-header {
        text-align: center;
        color: #1e3a8a;
        padding: 1.5rem 0;
        font-size: 2.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton>button {
        width: 100%;
    }
    
    /* Selectbox styling */
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
st.markdown("<h1 class='main-header'>🎯 HỆ THỐNG PHÁT HIỆN VẬT THỂ</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    # Logo
    st.markdown("""
        <div class='logo-container'>
            <h1 style='color: white; margin: 0; font-size: 3rem;'>🎯</h1>
            <p style='color: white; margin: 0.5rem 0 0 0; font-size: 1.2rem;'>Object Detection</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Tiêu đề nhóm
    st.markdown("""
        <div class='group-title'>
            📚 NHÓM 12<br>
            <span style='font-size: 0.9rem;'>Phát hiện vật thể</span>
        </div>
    """, unsafe_allow_html=True)
    
    # Navigation menu
    st.markdown("<p style='color: white; font-weight: bold; font-size: 1.1rem; margin-top: 1rem;'>🧭 CHỨC NĂNG</p>", unsafe_allow_html=True)
    
    option = st.selectbox(
        "Chọn chức năng:",
        ["🖼️ Phát hiện từ ảnh", "🎥 Phát hiện từ video", "📊 Phân tích model", "📈 Visualize Results"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Thông tin nhóm
    with st.expander("👥 Thành viên nhóm", expanded=False):
        st.markdown("""
        <div style='color: white;'>
        • Thành viên 1<br>
        • Thành viên 2<br>
        • Thành viên 3<br>
        • Thành viên 4
        </div>
        """, unsafe_allow_html=True)
    
    # Hướng dẫn
    with st.expander("📖 Hướng dẫn sử dụng", expanded=False):
        st.markdown("""
        <div style='color: white;'>
        <b>🖼️ Phát hiện từ ảnh:</b><br>
        Upload một hoặc nhiều ảnh để phát hiện vật thể<br><br>
        
        <b>🎥 Phát hiện từ video:</b><br>
        Upload video để phát hiện và theo dõi vật thể<br><br>
        
        <b>📊 Phân tích model:</b><br>
        Đánh giá hiệu suất model với file CSV<br><br>
        
        <b>📈 Visualize Results:</b><br>
        Xem các biểu đồ confusion matrix và kết quả training
        </div>
        """, unsafe_allow_html=True)

# -------------------------
# ẢNH
# -------------------------
if option == "🖼️ Phát hiện từ ảnh":
    st.header("📷 Phát hiện động vật từ ảnh")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🖼️ Upload ảnh")
        upload_files = st.file_uploader(
            "Chọn một hoặc nhiều ảnh", 
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Hỗ trợ định dạng: JPG, JPEG, PNG"
        )
    
    if upload_files:
        for idx, upload in enumerate(upload_files):
            st.markdown(f"### 🖼️ Ảnh {idx + 1}: {upload.name}")
            
            col_left, col_right = st.columns(2)
            
            try:
                file_bytes = upload.read()
                import numpy as np
                import cv2
                
                # Đọc ảnh
                img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
                
                if img is None:
                    st.error(f"❌ Không thể đọc ảnh {upload.name}")
                    continue
                
                with col_left:
                    st.markdown("**Ảnh gốc**")
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # Detect
                with st.spinner(f"🔍 Đang phát hiện động vật trong {upload.name}..."):
                    annotated, class_count = detect_image(img)
                
                with col_right:
                    st.markdown("**Kết quả phát hiện**")
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # Thống kê
                if class_count:
                    st.success("✅ Phát hiện thành công!")
                    
                    # Kiểm tra kiểu dữ liệu của class_count
                    if isinstance(class_count, dict) and class_count:
                        with st.expander("📊 Thống kê số lượng động vật"):
                            stats_col1, stats_col2 = st.columns(2)
                            with stats_col1:
                                for animal, count in class_count.items():
                                    st.metric(label=str(animal).capitalize(), value=count)
                            with stats_col2:
                                st.bar_chart(class_count)
                    elif isinstance(class_count, (int, float)):
                        st.info(f"📊 Tổng số đối tượng phát hiện: {class_count}")
                    else:
                        st.warning("⚠️ Không có thông tin thống kê chi tiết")
                else:
                    st.warning("⚠️ Không phát hiện được động vật nào trong ảnh")
                
                st.markdown("---")
                
            except Exception as e:
                st.error(f"❌ Lỗi xử lý ảnh {upload.name}: {str(e)}")
    else:
        st.info("👆 Vui lòng upload ảnh để bắt đầu phát hiện")

# -------------------------
# VIDEO
# -------------------------
elif option == "🎥 Phát hiện từ video":
    st.header("🎥 Phát hiện động vật từ video")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📹 Upload video")
        upload_files = st.file_uploader(
            "Chọn một hoặc nhiều video", 
            type=["mp4", "avi", "mov"],
            accept_multiple_files=True,
            help="Hỗ trợ định dạng: MP4, AVI, MOV"
        )
    
    if upload_files:
        for idx, upload in enumerate(upload_files):
            st.markdown(f"### 🎬 Video {idx + 1}: {upload.name}")
            
            col_left, col_right = st.columns(2)
            
            try:
                # Lưu video tạm
                temp_input = f"temp_input_{idx}.mp4"
                with open(temp_input, "wb") as f:
                    f.write(upload.read())
                
                with col_left:
                    st.markdown("**Video gốc**")
                    st.video(temp_input)
                
                # Detect
                with st.spinner(f"🔍 Đang phát hiện động vật trong {upload.name}... (có thể mất vài phút)"):
                    output_path, class_count = detect_video(temp_input)
                
                with col_right:
                    st.markdown("**Kết quả phát hiện**")
                    if os.path.exists(output_path):
                        st.video(output_path)
                    else:
                        st.error("❌ Không tìm thấy video kết quả")
                
                # Thống kê
                if class_count:
                    st.success("✅ Xử lý video thành công!")
                    
                    # Kiểm tra kiểu dữ liệu của class_count
                    if isinstance(class_count, dict) and class_count:
                        with st.expander("📊 Thống kê số lượng động vật xuất hiện"):
                            stats_col1, stats_col2 = st.columns(2)
                            with stats_col1:
                                for animal, count in class_count.items():
                                    st.metric(label=str(animal).capitalize(), value=count)
                            with stats_col2:
                                st.bar_chart(class_count)
                    elif isinstance(class_count, (int, float)):
                        st.info(f"📊 Tổng số đối tượng phát hiện: {class_count}")
                    else:
                        st.warning("⚠️ Không có thông tin thống kê chi tiết")
                    
                    # Tải xuống
                    if os.path.exists(output_path):
                        with open(output_path, "rb") as file:
                            st.download_button(
                                label="⬇️ Tải video đã xử lý",
                                data=file,
                                file_name=f"detected_{upload.name}",
                                mime="video/mp4"
                            )
                else:
                    st.warning("⚠️ Không phát hiện được động vật nào trong video")
                
                # Xóa file tạm
                if os.path.exists(temp_input):
                    os.remove(temp_input)
                
                st.markdown("---")
                
            except Exception as e:
                st.error(f"❌ Lỗi xử lý video {upload.name}: {str(e)}")
                if os.path.exists(temp_input):
                    os.remove(temp_input)
    else:
        st.info("👆 Vui lòng upload video để bắt đầu phát hiện")

# -------------------------
# PHÂN TÍCH MODEL
# -------------------------
elif option == "📊 Phân tích model":
    st.header("📈 Phân tích hiệu suất model")

    st.info("""
    📋 **Yêu cầu định dạng file CSV:**
    - Phải có 2 cột: `y_true` (nhãn thực tế) và `y_pred` (nhãn dự đoán)
    - Ví dụ:
    ```
    y_true,y_pred
    cat,cat
    dog,dog
    cat,dog
    bird,bird
    ```
    """)

    file = st.file_uploader("📂 Upload file CSV", type=["csv"])

    if file:
        try:
            import pandas as pd
            df = pd.read_csv(file)
            
            # Hiển thị preview
            with st.expander("👀 Preview dữ liệu", expanded=True):
                st.write(f"**Số dòng:** {len(df)} | **Số cột:** {len(df.columns)}")
                st.write("**Tên các cột:**", list(df.columns))
                st.dataframe(df.head(10), use_container_width=True)
            
            # Kiểm tra cột
            if 'y_true' not in df.columns or 'y_pred' not in df.columns:
                st.error("❌ File CSV phải có 2 cột: `y_true` và `y_pred`")
                
                # Cho phép người dùng chọn cột
                st.warning("💡 Hoặc chọn cột phù hợp từ dữ liệu của bạn:")
                col1, col2 = st.columns(2)
                with col1:
                    true_col = st.selectbox("Chọn cột nhãn thực tế:", df.columns, key="true")
                with col2:
                    pred_col = st.selectbox("Chọn cột nhãn dự đoán:", df.columns, key="pred")
                
                if st.button("🚀 Phân tích với các cột đã chọn"):
                    y_true = df[true_col]
                    y_pred = df[pred_col]
                else:
                    st.stop()
            else:
                y_true = df["y_true"]
                y_pred = df["y_pred"]
            
            # Hiển thị phân tích
            st.success("✅ Dữ liệu hợp lệ! Đang phân tích...")
            
            tab1, tab2, tab3 = st.tabs(["📊 Confusion Matrix", "📈 Classification Report", "📉 Metrics Summary"])
            
            with tab1:
                st.subheader("Ma trận nhầm lẫn (Confusion Matrix)")
                try:
                    fig = generate_confusion_matrix(y_true, y_pred, class_names=sorted(y_true.unique()))
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Lỗi tạo confusion matrix: {str(e)}")
            
            with tab2:
                st.subheader("Báo cáo phân loại (Classification Report)")
                try:
                    report = report_text(y_true, y_pred)
                    st.text(report)
                except Exception as e:
                    st.error(f"Lỗi tạo classification report: {str(e)}")
            
            with tab3:
                st.subheader("Tổng quan các chỉ số")
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                col1, col2, col3, col4 = st.columns(4)
                
                try:
                    with col1:
                        acc = accuracy_score(y_true, y_pred)
                        st.metric("Accuracy", f"{acc:.2%}")
                    
                    with col2:
                        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                        st.metric("Precision", f"{prec:.2%}")
                    
                    with col3:
                        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                        st.metric("Recall", f"{rec:.2%}")
                    
                    with col4:
                        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                        st.metric("F1-Score", f"{f1:.2%}")
                except Exception as e:
                    st.error(f"Lỗi tính toán metrics: {str(e)}")
            
        except Exception as e:
            st.error(f"❌ Lỗi đọc file CSV: {str(e)}")
            st.info("💡 Vui lòng kiểm tra lại định dạng file CSV")
    else:
        st.info("👆 Vui lòng upload file CSV để bắt đầu phân tích")

# -------------------------
# VISUALIZE RESULTS
# -------------------------
elif option == "📈 Visualize Results":
    st.header("📈 Trực quan hóa kết quả Training")
    
    tab1, tab2, tab3 = st.tabs(["📊 Confusion Matrix", "📉 Training Curves", "🎯 Class Distribution"])
    
    with tab1:
        st.subheader("Ma trận nhầm lẫn (Confusion Matrix)")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### 📂 Upload Confusion Matrix")
            cm_file = st.file_uploader(
                "Upload ảnh confusion matrix",
                type=["png", "jpg", "jpeg"],
                key="cm_upload",
                help="Upload ảnh confusion matrix từ folder results"
            )
            
            if cm_file:
                import cv2
                import numpy as np
                file_bytes = cm_file.read()
                img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                with col2:
                    st.image(img_rgb, caption="Confusion Matrix", use_container_width=True)
            else:
                with col2:
                    st.info("👈 Vui lòng upload ảnh confusion matrix")
        
        st.markdown("---")
        
        # Normalized confusion matrix
        st.markdown("#### 📊 Normalized Confusion Matrix")
        
        col3, col4 = st.columns([1, 2])
        
        with col3:
            norm_cm_file = st.file_uploader(
                "Upload normalized confusion matrix",
                type=["png", "jpg", "jpeg"],
                key="norm_cm_upload",
                help="Upload ảnh normalized confusion matrix"
            )
            
            if norm_cm_file:
                import cv2
                import numpy as np
                file_bytes = norm_cm_file.read()
                img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                with col4:
                    st.image(img_rgb, caption="Normalized Confusion Matrix", use_container_width=True)
            else:
                with col4:
                    st.info("👈 Vui lòng upload ảnh normalized confusion matrix")
    
    with tab2:
        st.subheader("📉 Đường cong Training")
        
        # Upload results.png hoặc nhiều ảnh training curves
        st.markdown("#### 📊 Training/Validation Curves")
        
        results_file = st.file_uploader(
            "Upload ảnh kết quả training (results.png)",
            type=["png", "jpg", "jpeg"],
            key="results_upload",
            help="Upload file results.png từ thư mục runs/detect/train"
        )
        
        if results_file:
            import cv2
            import numpy as np
            file_bytes = results_file.read()
            img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            st.image(img_rgb, caption="Training Results", use_container_width=True)
            
            # Phân tích
            with st.expander("📊 Phân tích kết quả", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("""
                    **🎯 Metrics cần chú ý:**
                    - **mAP50**: Mean Average Precision @ IoU 0.5
                    - **mAP50-95**: mAP trung bình từ IoU 0.5-0.95
                    - **Precision**: Độ chính xác dự đoán
                    - **Recall**: Khả năng phát hiện đối tượng
                    """)
                
                with col2:
                    st.markdown("""
                    **📉 Loss Functions:**
                    - **Box Loss**: Lỗi dự đoán bounding box
                    - **Class Loss**: Lỗi phân loại
                    - **DFL Loss**: Distribution Focal Loss
                    """)
                
                with col3:
                    st.markdown("""
                    **✅ Dấu hiệu model tốt:**
                    - Loss giảm dần theo epoch
                    - mAP tăng dần và ổn định
                    - Không có dấu hiệu overfitting
                    - Val loss gần train loss
                    """)
        else:
            st.info("👆 Vui lòng upload file results.png để xem đường cong training")
        
        st.markdown("---")
        
        # Upload thêm các biểu đồ khác
        st.markdown("#### 📈 Các biểu đồ khác")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pr_curve = st.file_uploader(
                "Upload PR Curve (Precision-Recall)",
                type=["png", "jpg", "jpeg"],
                key="pr_upload"
            )
            
            if pr_curve:
                import cv2
                import numpy as np
                file_bytes = pr_curve.read()
                img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption="PR Curve", use_container_width=True)
        
        with col2:
            f1_curve = st.file_uploader(
                "Upload F1 Curve",
                type=["png", "jpg", "jpeg"],
                key="f1_upload"
            )
            
            if f1_curve:
                import cv2
                import numpy as np
                file_bytes = f1_curve.read()
                img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption="F1 Curve", use_container_width=True)
    
    with tab3:
        st.subheader("🎯 Phân bố Class và Labels")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Label Distribution")
            labels_file = st.file_uploader(
                "Upload ảnh labels distribution",
                type=["png", "jpg", "jpeg"],
                key="labels_upload"
            )
            
            if labels_file:
                import cv2
                import numpy as np
                file_bytes = labels_file.read()
                img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption="Labels Distribution", use_container_width=True)
        
        with col2:
            st.markdown("#### 🖼️ Train Batch Examples")
            batch_file = st.file_uploader(
                "Upload ảnh train batch",
                type=["png", "jpg", "jpeg"],
                key="batch_upload"
            )
            
            if batch_file:
                import cv2
                import numpy as np
                file_bytes = batch_file.read()
                img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption="Train Batch", use_container_width=True)
        
        st.markdown("---")
        
        # Predictions examples
        st.markdown("#### 🎯 Validation Predictions")
        
        pred_files = st.file_uploader(
            "Upload ảnh val predictions (có thể chọn nhiều)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="pred_upload"
        )
        
        if pred_files:
            cols = st.columns(3)
            for idx, pred_file in enumerate(pred_files):
                import cv2
                import numpy as np
                file_bytes = pred_file.read()
                img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                with cols[idx % 3]:
                    st.image(img_rgb, caption=f"Prediction {idx+1}", use_container_width=True)
        else:
            st.info("👆 Upload các ảnh validation predictions để xem kết quả dự đoán")
    
    # Hướng dẫn
    with st.expander("📖 Hướng dẫn tìm các file results", expanded=False):
        st.markdown("""
        ### 📁 Vị trí các file sau khi training YOLOv8:
        
        Sau khi training xong, các file kết quả thường nằm trong thư mục:
        ```
        runs/detect/train/
        ├── confusion_matrix.png
        ├── confusion_matrix_normalized.png
        ├── results.png
        ├── PR_curve.png
        ├── F1_curve.png
        ├── labels.jpg
        ├── train_batch0.jpg
        ├── val_batch0_labels.jpg
        └── val_batch0_pred.jpg
        ```
        
        ### 📊 Ý nghĩa các file:
        
        - **confusion_matrix.png**: Ma trận nhầm lẫn
        - **results.png**: Tổng hợp các metrics theo epoch
        - **PR_curve.png**: Đường cong Precision-Recall
        - **F1_curve.png**: Đường cong F1-Score
        - **labels.jpg**: Phân bố nhãn trong dataset
        - **train_batch0.jpg**: Ví dụ các ảnh training
        - **val_batch0_pred.jpg**: Kết quả dự đoán trên validation set
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🐾 Animal Detection App | Powered by YOLOv8 & Streamlit</p>
</div>
""", unsafe_allow_html=True)