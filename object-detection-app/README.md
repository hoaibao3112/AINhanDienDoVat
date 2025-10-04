# 🤖 Ứng dụng AI Nhận diện Đồ vật

Ứng dụng AI sử dụng camera laptop để nhận diện đồ vật đặt trước camera với YOLOv8.

## ✨ Tính năng

- 🎯 **Nhận diện đồ vật thời gian thực** sử dụng YOLOv8 (pre-trained COCO dataset)
- 📊 **Đếm số lượng** từng loại đồ vật
- 🎨 **Hiển thị bounding box** và tên đồ vật
- 📈 **Bảng thống kê** số lượng đối tượng hiện tại
- 💾 **Lưu log** vào file CSV với thời gian
- 📸 **Chụp ảnh màn hình** kết quả nhận diện

## 🎯 Đối tượng có thể nhận diện

Ứng dụng tập trung vào các đồ vật thông thường trong văn phòng và gia đình:
- 📚 **Văn phòng phẩm**: book, laptop, mouse, keyboard, cell phone
- 🍽️ **Đồ dùng ăn uống**: bottle, cup, wine glass, fork, knife, spoon, bowl
- 🪑 **Nội thất**: chair, couch, bed, dining table
- 🧸 **Đồ chơi & phụ kiện**: teddy bear, handbag, tie, suitcase
- 🍎 **Thực phẩm**: banana, apple, sandwich, orange, pizza, cake
- Và nhiều đối tượng khác từ COCO dataset...

## 📋 Yêu cầu hệ thống

- **Python**: 3.8 trở lên
- **Hệ điều hành**: Windows 10/11, macOS, Linux
- **Camera**: Webcam hoặc camera laptop
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB+)
- **GPU**: Không bắt buộc (CPU cũng chạy được, GPU sẽ nhanh hơn)

## 🚀 Cài đặt và Sử dụng

### Bước 1: Cài đặt dependencies
```bash
# Chạy file batch (Windows)
install_requirements.bat

# Hoặc cài đặt thủ công
pip install -r requirements.txt
```

### Bước 2: Kiểm tra hệ thống
```bash
python check_system.py
```

### Bước 3: Test YOLOv8 model
```bash
python test_yolo.py
```

### Bước 4: Chạy ứng dụng
```bash
# Cách 1: Chạy trực tiếp
python object_detection_app.py

# Cách 2: Sử dụng PowerShell launcher (Windows)
./run_app.ps1
```

## 🎮 Hướng dẫn sử dụng

Khi ứng dụng đang chạy:
- **'q'** hoặc **'ESC'**: Thoát ứng dụng
- **'s'**: Chụp ảnh màn hình với kết quả nhận diện
- **'r'**: Reset bộ đếm về 0
- **'l'**: Lưu log ngay lập tức vào file CSV

## 📊 File Output

### Screenshots
- Định dạng: `screenshot_YYYYMMDD_HHMMSS.jpg`
- Chứa: Frame với bounding box và thống kê

### Detection Logs  
- Định dạng: `detection_log_YYYYMMDD_HHMMSS.csv`
- Cột: `timestamp`, `object_name`, `count`
- Tự động lưu mỗi 5 giây và khi thoát ứng dụng

## 🏗️ Cấu trúc dự án

```
object-detection-app/
├── 📄 object_detection_app.py    # Ứng dụng chính
├── 🧪 test_yolo.py              # Test YOLOv8 model  
├── 🔍 check_system.py           # Kiểm tra hệ thống
├── 📦 requirements.txt          # Dependencies
├── 🔨 install_requirements.bat  # Script cài đặt (Windows)
├── 🚀 run_app.ps1              # PowerShell launcher
├── 📚 README.md                # File này
├── 📊 detection_log_*.csv      # Log files (tự tạo)
└── 📸 screenshot_*.jpg         # Screenshots (tự tạo)
```

## 🛠️ Troubleshooting

### Camera không hoạt động
```bash
# Kiểm tra camera
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error')"
```

### YOLOv8 model không tải được
```bash
# Tải lại model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Lỗi OpenCV
```bash
# Cài đặt lại OpenCV
pip uninstall opencv-python
pip install opencv-python==4.8.1.78
```

### Performance thấp
- Giảm độ phân giải camera trong code
- Sử dụng YOLOv8n (nano) thay vì YOLOv8s/m/l
- Tăng interval xử lý frame

## 🎨 Tùy chỉnh

### Thay đổi model YOLOv8
```python
# Trong object_detection_app.py, dòng 47
self.model = YOLO('yolov8n.pt')  # nano (nhanh)
# self.model = YOLO('yolov8s.pt')  # small
# self.model = YOLO('yolov8m.pt')  # medium  
# self.model = YOLO('yolov8l.pt')  # large (chậm nhưng chính xác)
```

### Thêm đối tượng quan tâm
```python
# Trong __init__(), thêm vào target_classes
self.target_classes = {
    'book', 'laptop', 'bottle', 'cup', 'cell phone',
    'your_new_object'  # Thêm đối tượng mới
}
```

### Thay đổi confidence threshold
```python
# Trong draw_predictions(), dòng 94
if class_name in self.target_classes and confidence > 0.5:  # Thay đổi 0.5
```

## 📈 Performance

### YOLOv8n (Nano)
- **Tốc độ**: ~30-60 FPS (CPU), ~100+ FPS (GPU)
- **Kích thước**: ~6MB
- **mAP**: 37.3%

### YOLOv8s (Small)  
- **Tốc độ**: ~20-40 FPS (CPU), ~80+ FPS (GPU)
- **Kích thước**: ~22MB
- **mAP**: 44.9%

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📝 License

Dự án này sử dụng MIT License. Xem file `LICENSE` để biết thêm chi tiết.

## 🙏 Acknowledgments

- **Ultralytics**: YOLOv8 framework
- **OpenCV**: Computer vision library  
- **COCO Dataset**: Pre-trained object classes
- **PyTorch**: Deep learning framework

## 📞 Liên hệ

Nếu có vấn đề hoặc đóng góp ý kiến, vui lòng tạo Issue trên GitHub.

---

*Được phát triển với ❤️ bởi AI Assistant*
# Bước 1: Chuyển vào thư mục dự án
cd "C:\Users\PC\Desktop\AI\object-detection-app"

# Bước 2: Chạy ứng dụng
C:/Users/PC/Desktop/AI/.venv/Scripts/python.exe object_detection_app.py