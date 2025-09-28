#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ứng dụng AI nhận diện đồ vật sử dụng YOLOv8
Tác giả: AI Assistant
Ngày: 28/09/2025
"""

import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import os
from collections import defaultdict, Counter
import time
from PIL import Image, ImageDraw, ImageFont
import sys

class ObjectDetectionApp:
    def __init__(self):
        """Khởi tạo ứng dụng"""
        self.model = None
        self.cap = None
        self.is_running = False
        self.log_data = []
        self.object_counts = defaultdict(int)
        self.detection_history = defaultdict(list)  # Lịch sử nhận diện để lọc nhiễu
        self.max_history = 10  # Giữ 10 frame gần nhất
        
        # Từ điển dịch tên đối tượng sang tiếng Việt (tên đầy đủ cho CSV)
        self.vietnamese_names = {
            'person': 'Nguoi',
            'bicycle': 'Xe dap', 
            'car': 'O to',
            'motorcycle': 'Xe may',
            'airplane': 'May bay',
            'bus': 'Xe buyt',
            'train': 'Tau hoa',
            'truck': 'Xe tai',
            'boat': 'Thuyen',
            'traffic light': 'Den giao thong',
            'stop sign': 'Bien bao dung',
            'bench': 'Bang ghe',
            'bird': 'Chim',
            'cat': 'Meo',
            'dog': 'Cho',
            'horse': 'Ngua',
            'sheep': 'Cuu', 
            'cow': 'Bo',
            'elephant': 'Voi',
            'bear': 'Gau',
            'zebra': 'Ngua van',
            'giraffe': 'Huong cao',
            'backpack': 'Ba lo',
            'umbrella': 'O',
            'handbag': 'Tui xach',
            'tie': 'Ca vat',
            'suitcase': 'Vali',
            'frisbee': 'Dia bay',
            'skis': 'Van truot tuyet',
            'snowboard': 'Van truot tuyet',
            'sports ball': 'Bong the thao',
            'kite': 'Dieu',
            'baseball bat': 'Gay bong chay',
            'baseball glove': 'Gang tay bong chay',
            'skateboard': 'Van truot',
            'surfboard': 'Van luot song',
            'tennis racket': 'Vot tennis',
            'bottle': 'Chai',
            'wine glass': 'Ly ruou vang',
            'cup': 'Coc',
            'fork': 'Nia',
            'knife': 'Dao',
            'spoon': 'Muong',
            'bowl': 'Bat',
            'banana': 'Chuoi',
            'apple': 'Tao',
            'sandwich': 'Banh sandwich',
            'orange': 'Cam',
            'broccoli': 'Sup lo xanh',
            'carrot': 'Ca rot',
            'hot dog': 'Banh mi kep xuc xich',
            'pizza': 'Pizza',
            'donut': 'Banh donut',
            'cake': 'Banh ngot',
            'chair': 'Ghe',
            'couch': 'Ghe sofa',
            'potted plant': 'Chau cay',
            'bed': 'Giuong',
            'dining table': 'Ban an',
            'toilet': 'Nha ve sinh',
            'tv': 'TV',
            'laptop': 'Laptop',
            'mouse': 'Chuot may tinh',
            'remote': 'Dieu khien tu xa',
            'keyboard': 'Ban phim',
            'cell phone': 'Dien thoai',
            'microwave': 'Lo vi song',
            'oven': 'Lo nuong',
            'toaster': 'May nuong banh mi',
            'sink': 'Bon rua',
            'refrigerator': 'Tu lanh',
            'book': 'Sach',
            'clock': 'Dong ho',
            'vase': 'Binh hoa',
            'scissors': 'Keo',
            'teddy bear': 'Gau bong',
            'hair drier': 'May say toc',
            'toothbrush': 'Ban chai danh rang'
        }
        
        # Từ điển tên ngắn gọn cho hiển thị (tránh lỗi font)
        self.display_names = {
            'person': 'Nguoi',
            'bicycle': 'Xe dap', 
            'car': 'Oto',
            'motorcycle': 'Xe may',
            'airplane': 'May bay',
            'bus': 'Xe buyt',
            'train': 'Tau hoa',
            'truck': 'Xe tai',
            'boat': 'Thuyen',
            'traffic light': 'Den giao thong',
            'stop sign': 'Bien bao',
            'bench': 'Bang ghe',
            'bird': 'Chim',
            'cat': 'Meo',
            'dog': 'Cho',
            'horse': 'Ngua',
            'sheep': 'Cuu', 
            'cow': 'Bo',
            'elephant': 'Voi',
            'bear': 'Gau',
            'zebra': 'Ngua van',
            'giraffe': 'Cao co',
            'backpack': 'Balo',
            'umbrella': 'O',
            'handbag': 'Tui xach',
            'tie': 'Ca vat',
            'suitcase': 'Vali',
            'frisbee': 'Dia bay',
            'skis': 'Van truot',
            'snowboard': 'Van truot',
            'sports ball': 'Bong',
            'kite': 'Dieu',
            'baseball bat': 'Gay bong',
            'baseball glove': 'Gang tay',
            'skateboard': 'Van truot',
            'surfboard': 'Van luot',
            'tennis racket': 'Vot',
            'bottle': 'Chai',
            'wine glass': 'Ly ruou',
            'cup': 'Coc',
            'fork': 'Nia',
            'knife': 'Dao',
            'spoon': 'Muong',
            'bowl': 'Bat',
            'banana': 'Chuoi',
            'apple': 'Tao',
            'sandwich': 'Sandwich',
            'orange': 'Cam',
            'broccoli': 'Sup lo',
            'carrot': 'Ca rot',
            'hot dog': 'Hot dog',
            'pizza': 'Pizza',
            'donut': 'Donut',
            'cake': 'Banh',
            'chair': 'Ghe',
            'couch': 'Sofa',
            'potted plant': 'Cay',
            'bed': 'Giuong',
            'dining table': 'Ban an',
            'toilet': 'WC',
            'tv': 'TV',
            'laptop': 'Laptop',
            'mouse': 'Chuot',
            'remote': 'Remote',
            'keyboard': 'Ban phim',
            'cell phone': 'Dien thoai',
            'microwave': 'Lo vi song',
            'oven': 'Lo nuong',
            'toaster': 'May nuong',
            'sink': 'Bon rua',
            'refrigerator': 'Tu lanh',
            'book': 'Sach',
            'clock': 'Dong ho',
            'vase': 'Binh hoa',
            'scissors': 'Keo',
            'teddy bear': 'Gau bong',
            'hair drier': 'May say',
            'toothbrush': 'Ban chai'
        }
        
        # Các đối tượng quan tâm từ COCO dataset
        self.target_classes = set(self.vietnamese_names.keys())
        
        print("🚀 Đang khởi tạo Object Detection App...")
        
        # Confidence thresholds riêng cho từng loại đối tượng
        self.confidence_thresholds = {
            # Đối tượng dễ nhầm lẫn - yêu cầu confidence cao hơn
            'knife': 0.85,
            'fork': 0.85,
            'spoon': 0.85,
            'hot dog': 0.85,
            'sandwich': 0.80,
            'donut': 0.80,
            'cake': 0.80,
            'pizza': 0.80,
            
            # Đối tượng rõ ràng - confidence trung bình
            'person': 0.60,
            'car': 0.65,
            'bicycle': 0.65,
            'motorcycle': 0.65,
            
            # Đối tượng tech - confidence trung bình cao
            'laptop': 0.70,
            'cell phone': 0.75,
            'tv': 0.70,
            'mouse': 0.75,
            'keyboard': 0.75,
            'remote': 0.75,
            
            # Đối tượng văn phòng phẩm
            'book': 0.70,
            'scissors': 0.80,
            
            # Mặc định cho các đối tượng khác
            'default': 0.70
        }
    
    def get_confidence_threshold(self, class_name):
        """Lấy confidence threshold phù hợp cho từng loại đối tượng"""
        return self.confidence_thresholds.get(class_name, self.confidence_thresholds['default'])
    
    def is_stable_detection(self, class_name, confidence):
        """Kiểm tra xem nhận diện có ổn định không (lọc nhiễu)"""
        # Thêm vào lịch sử
        self.detection_history[class_name].append(confidence)
        
        # Giữ chỉ max_history frame gần nhất
        if len(self.detection_history[class_name]) > self.max_history:
            self.detection_history[class_name].pop(0)
        
        # Yêu cầu ít nhất 3 frame liên tiếp để xác nhận
        if len(self.detection_history[class_name]) < 3:
            return False
        
        # Kiểm tra 3 frame gần nhất có confidence > threshold không
        recent_detections = self.detection_history[class_name][-3:]
        threshold = self.get_confidence_threshold(class_name)
        
        stable_count = sum(1 for conf in recent_detections if conf > threshold)
        return stable_count >= 2  # Ít nhất 2/3 frame gần nhất phải > threshold
    
    def get_vietnamese_names(self, english_name):
        """Lấy tên tiếng Việt cho hiển thị và lưu CSV"""
        vietnamese_name = self.vietnamese_names.get(english_name, english_name)  # Cho CSV
        display_name = self.display_names.get(english_name, english_name)        # Cho hiển thị
        return vietnamese_name, display_name
        
    def load_model(self):
        """Tải YOLOv8 model"""
        try:
            print("📦 Đang tải YOLOv8 model...")
            self.model = YOLO('yolov8n.pt')  # YOLOv8 nano - nhanh và nhẹ
            print("✅ Model đã được tải thành công!")
            return True
        except Exception as e:
            print(f"❌ Lỗi khi tải model: {e}")
            return False
    
    def init_camera(self, camera_id=0):
        """Khởi tạo camera"""
        try:
            print(f"📹 Đang khởi tạo camera {camera_id}...")
            self.cap = cv2.VideoCapture(camera_id)
            
            if not self.cap.isOpened():
                raise Exception("Không thể mở camera")
            
            # Thiết lập độ phân giải
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            print("✅ Camera đã sẵn sàng!")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi camera: {e}")
            return False
    
    def draw_predictions(self, frame, results):
        """Vẽ bounding box và label lên frame"""
        current_objects = Counter()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Lấy tọa độ và confidence
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    class_id = int(box.cls[0])
                    
                    # Lấy tên class
                    class_name = self.model.names[class_id]
                    
                    # Chỉ xử lý các đối tượng quan tâm
                    if class_name in self.target_classes:
                        confidence_threshold = self.get_confidence_threshold(class_name)
                        vietnamese_name, display_name = self.get_vietnamese_names(class_name)
                        
                        # Kiểm tra confidence và độ ổn định
                        if (confidence > confidence_threshold and 
                            self.is_stable_detection(class_name, confidence)):
                            # Nhận diện chính thức - màu xanh
                            current_objects[vietnamese_name] += 1
                            box_color = (0, 255, 0)  # Xanh lá
                            text_bg_color = (0, 255, 0)
                            status = "OK"
                        elif confidence > 0.5:
                            # Nghi ngờ - màu vàng
                            box_color = (0, 255, 255)  # Vàng
                            text_bg_color = (0, 255, 255)
                            status = "?"
                        else:
                            # Bỏ qua confidence quá thấp
                            continue
                        
                        # Vẽ bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                        
                        # Vẽ label với thông tin chi tiết
                        label = f"{display_name}: {confidence:.2f} {status}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                        
                        # Background cho text
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0] + 10, y1), text_bg_color, -1)
                        
                        # Text với font an toàn
                        cv2.putText(frame, label, (x1 + 2, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        return frame, current_objects
    
    def draw_counter_panel(self, frame, object_counts):
        """Vẽ bảng đếm đối tượng"""
        panel_height = min(200, len(object_counts) * 25 + 50)
        panel = np.zeros((panel_height, 350, 3), dtype=np.uint8)
        
        # Header bằng tiếng Việt (dùng ký tự an toàn)
        cv2.putText(panel, "BANG THONG KE", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.line(panel, (10, 35), (340, 35), (255, 255, 255), 1)
        
        # Danh sách đối tượng
        y_offset = 55
        total_objects = 0
        
        for obj_name, count in object_counts.items():
            if count > 0:
                # Tìm tên hiển thị ngắn gọn từ tên đầy đủ
                display_name = obj_name
                for eng_name, viet_name in self.vietnamese_names.items():
                    if viet_name == obj_name:
                        display_name = self.display_names.get(eng_name, obj_name)
                        break
                
                text = f"{display_name}: {count}"
                cv2.putText(panel, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_offset += 25
                total_objects += count
        
        # Tổng cộng
        cv2.line(panel, (10, y_offset), (340, y_offset), (255, 255, 255), 1)
        cv2.putText(panel, f"TONG CONG: {total_objects}", (10, y_offset + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Ghép panel vào frame
        frame_height, frame_width = frame.shape[:2]
        panel_resized = cv2.resize(panel, (min(350, frame_width//3), 
                                         min(panel_height, frame_height//2)))
        
        h, w = panel_resized.shape[:2]
        frame[10:10+h, frame_width-w-10:frame_width-10] = panel_resized
        
        return frame
    
    def log_detection(self, object_counts):
        """Ghi log vào file CSV"""
        if any(count > 0 for count in object_counts.values()):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            for obj_name, count in object_counts.items():
                if count > 0:
                    self.log_data.append({
                        'timestamp': timestamp,
                        'object_name': obj_name,  # Đã là tên tiếng Việt
                        'count': count
                    })
    
    def save_log_to_csv(self):
        """Lưu log vào file CSV"""
        if self.log_data:
            df = pd.DataFrame(self.log_data)
            # Thay đổi header thành tiếng Việt
            df.columns = ['Thoi_gian', 'Ten_do_vat', 'So_luong']
            filename = f"nhan_dien_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False, encoding='utf-8-sig')  # utf-8-sig để hiển thị đúng tiếng Việt
            print(f"📊 Log đã được lưu vào {filename}")
    
    def run(self):
        """Chạy ứng dụng chính"""
        if not self.load_model():
            return False
        
        if not self.init_camera():
            return False
        
        self.is_running = True
        last_log_time = time.time()
        
        print("\n" + "="*50)
        print("🎯 ỨNG DỤNG NHẬN DIỆN ĐỒ VẬT")
        print("="*50)
        print("📋 Hướng dẫn sử dụng:")
        print("   • 'q' hoặc 'ESC': Thoát ứng dụng")
        print("   • 's': Chụp ảnh màn hình")
        print("   • 'r': Reset bộ đếm")
        print("   • 'l': Lưu log ngay lập tức")
        print("="*50)
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Không thể đọc frame từ camera")
                    break
                
                # Lật frame theo chiều ngang (mirror effect)
                frame = cv2.flip(frame, 1)
                
                # Thực hiện detection
                results = self.model(frame, verbose=False)
                
                # Vẽ predictions
                frame, current_objects = self.draw_predictions(frame, results)
                
                # Cập nhật object counts (sử dụng tên tiếng Việt)
                for vietnamese_name in current_objects:
                    if vietnamese_name in current_objects:
                        self.object_counts[vietnamese_name] = max(self.object_counts[vietnamese_name], 
                                                                current_objects[vietnamese_name])
                
                # Vẽ bảng đếm
                frame = self.draw_counter_panel(frame, self.object_counts)
                
                # Thông tin FPS
                cv2.putText(frame, f"FPS: {int(self.cap.get(cv2.CAP_PROP_FPS))}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Thông tin chế độ lọc
                cv2.putText(frame, "Xanh: Xac nhan | Vang: Nghi ngo", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Thêm hướng dẫn sử dụng (dùng ký tự an toàn)
                cv2.putText(frame, "Q: Thoat | S: Chup anh | R: Reset | L: Luu log", 
                           (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Hiển thị frame với tiêu đề tiếng Việt
                cv2.imshow('Ung dung AI Nhan dien Do vat - Nhan Q de thoat', frame)
                
                # Log định kỳ (mỗi 5 giây)
                if time.time() - last_log_time > 5:
                    self.log_detection(self.object_counts)
                    last_log_time = time.time()
                
                # Xử lý phím
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' hoặc ESC
                    break
                elif key == ord('s'):  # Chụp ảnh
                    screenshot_name = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(screenshot_name, frame)
                    print(f"📸 Đã chụp ảnh: {screenshot_name}")
                elif key == ord('r'):  # Reset counter
                    self.object_counts.clear()
                    self.detection_history.clear()  # Xóa luôn lịch sử nhận diện
                    print("🔄 Đã reset bộ đếm và lịch sử nhận diện")
                elif key == ord('l'):  # Lưu log
                    self.save_log_to_csv()
        
        except KeyboardInterrupt:
            print("\n⚠️  Người dùng dừng ứng dụng")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Dọn dẹp resources"""
        print("\n🧹 Đang dọn dẹp...")
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Lưu log cuối cùng
        if self.log_data:
            self.save_log_to_csv()
        
        print("✅ Đã thoát ứng dụng thành công!")

def main():
    """Hàm main"""
    app = ObjectDetectionApp()
    app.run()

if __name__ == "__main__":
    main()