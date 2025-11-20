import streamlit as st
from utils.detect import detect_image
from utils.video import detect_video
from utils.analysis import *
import os

st.set_page_config(page_title="Animal Detection App", layout="wide", page_icon="🐾")

# CSS tùy chỉnh
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        padding: 1rem 0;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🐾 Ứng dụng phát hiện động vật bằng YOLOv8</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/2E86AB/FFFFFF?text=Animal+Detection", use_container_width=True)
    option = st.selectbox(
        "🎯 Chọn chức năng",
        ["Detect Image", "Detect Video", "Model Analysis"]
    )
    st.markdown("---")
    st.markdown("""
    ### 📖 Hướng dẫn
    - **Detect Image**: Upload ảnh để phát hiện động vật
    - **Detect Video**: Upload video để phát hiện động vật
    - **Model Analysis**: Phân tích hiệu suất model
    """)

# -------------------------
# ẢNH
# -------------------------
if option == "Detect Image":
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
elif option == "Detect Video":
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
elif option == "Model Analysis":
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

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🐾 Animal Detection App | Powered by YOLOv8 & Streamlit</p>
</div>
""", unsafe_allow_html=True)