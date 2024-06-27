import cv2
import numpy as np
from PIL import Image
import streamlit as st

# Đường dẫn đến mô hình và tệp cấu hình
MODEL = "model/MobileNetSSD_deploy.caffemodel"
PROTOTXT = "model/MobileNetSSD_deploy.prototxt.txt"

def load_model(prototxt, model):
    """Hàm để tải mô hình."""
    return cv2.dnn.readNetFromCaffe(prototxt, model)

def process_image(image, net):
    """Hàm xử lý hình ảnh để phát hiện đối tượng."""
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    return detections

def annotate_image(image, detections, confidence_threshold=0.5):
    """Hàm chú thích hình ảnh với các khung bao quanh đối tượng được phát hiện."""
    (height, width) = image.shape[:2]

    for detection_index in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, detection_index, 2]

        if confidence > confidence_threshold:
            int(detections[0, 0, detection_index, 1])
            box = detections[0, 0, detection_index, 3:7] * np.array([width, height, width, height])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            # Đảm bảo tọa độ nằm trong giới hạn của hình ảnh
            start_x = max(0, start_x)
            start_y = max(0, start_y)
            end_x = min(width - 1, end_x)
            end_y = min(height - 1, end_y)

            cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (70, 2), 2)

    return image

def main():
    """Hàm chính để chạy ứng dụng Streamlit."""
    # Thiết lập giao diện Streamlit
    st.title('Object Detection for Images')
    file = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])
    
    # Kiểm tra nếu người dùng đã tải lên hình ảnh
    if file is not None:
        st.image(file, caption="Uploaded Image")

        image = Image.open(file)
        image = np.array(image)

        # Tải mô hình
        net = load_model(PROTOTXT, MODEL)
        detections = process_image(image, net)
        processed_image = annotate_image(image, detections)
        
        st.image(processed_image, caption="Processed Image")

# Chạy ứng dụng nếu tệp này được chạy như một chương trình chính
if __name__ == "__main__":
    main()