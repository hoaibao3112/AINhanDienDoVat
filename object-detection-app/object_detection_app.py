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
        self.max_history = 15  # Tăng lên 15 frame để học tốt hơn
        self.object_stability = defaultdict(int)  # Đếm độ ổn định của từng đối tượng
        self.false_positive_penalty = defaultdict(float)  # Phạt các detection hay sai
        self.learning_rate = 0.1  # Tốc độ học
        self.smart_mode = "normal"  # normal, strict, sensitive
        
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
        
        # Context rules - đối tượng thường xuất hiện cùng nhau
        self.context_rules = {
            'laptop': ['mouse', 'keyboard', 'cell phone'],  # Laptop thường đi với chuột, bàn phím
            'dining table': ['chair', 'cup', 'bowl', 'fork', 'knife', 'spoon'],  # Bàn ăn với đồ ăn
            'person': ['cell phone', 'handbag', 'book'],  # Người thường có điện thoại, túi
            'kitchen': ['refrigerator', 'microwave', 'sink', 'bottle'],
            'office': ['laptop', 'keyboard', 'mouse', 'book', 'scissors']
        }
        
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
    
    def is_reasonable_size(self, class_name, bbox_width, bbox_height, frame_width, frame_height):
        """Kiểm tra kích thước bounding box có hợp lý không"""
        bbox_area_ratio = (bbox_width * bbox_height) / (frame_width * frame_height)
        
        # Định nghĩa kích thước hợp lý cho từng loại đối tượng
        size_ranges = {
            # Đối tượng nhỏ (2-15% màn hình)
            'cell phone': (0.02, 0.15),
            'mouse': (0.01, 0.08),
            'remote': (0.02, 0.12),
            'book': (0.03, 0.20),
            'scissors': (0.01, 0.10),
            'knife': (0.005, 0.08),
            'fork': (0.005, 0.05),
            'spoon': (0.005, 0.05),
            'cup': (0.02, 0.15),
            'bottle': (0.02, 0.20),
            
            # Đối tượng trung bình (10-40% màn hình)
            'laptop': (0.10, 0.40),
            'keyboard': (0.08, 0.30),
            'tv': (0.15, 0.60),
            'chair': (0.15, 0.50),
            
            # Đối tượng lớn (20-80% màn hình)
            'person': (0.20, 0.80),
            'couch': (0.25, 0.70),
            'bed': (0.30, 0.80),
            'car': (0.25, 0.85),
            
            # Mặc định
            'default': (0.01, 0.60)
        }
        
        min_size, max_size = size_ranges.get(class_name, size_ranges['default'])
        return min_size <= bbox_area_ratio <= max_size
    
    def is_reasonable_position(self, class_name, x1, y1, x2, y2, frame_width, frame_height):
        """Kiểm tra vị trí đối tượng có hợp lý không"""
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Một số đối tượng thường xuất hiện ở vị trí đặc biệt
        position_rules = {
            'laptop': lambda cx, cy: cy > frame_height * 0.3,  # Laptop thường ở phần dưới
            'tv': lambda cx, cy: cy < frame_height * 0.7,      # TV thường ở phần trên/giữa
            'ceiling fan': lambda cx, cy: cy < frame_height * 0.3,  # Quạt trần ở trên
            'chair': lambda cx, cy: cy > frame_height * 0.2,   # Ghế thường ở phần dưới
        }
        
        if class_name in position_rules:
            return position_rules[class_name](center_x, center_y)
        
        return True  # Không có rule đặc biệt thì chấp nhận
    
    def get_context_boost(self, class_name, current_detections):
        """Tăng độ tin cậy dựa trên ngữ cảnh (đối tượng khác trong khung hình)"""
        boost = 0.0
        
        for main_object, related_objects in self.context_rules.items():
            if class_name in related_objects:
                # Nếu đối tượng chính xuất hiện, tăng độ tin cậy cho related objects
                if main_object in current_detections:
                    boost += 0.05
                    
        # Tăng độ tin cậy nếu có nhiều đối tượng cùng context
        context_count = 0
        for main_object, related_objects in self.context_rules.items():
            if class_name == main_object or class_name in related_objects:
                context_count += sum(1 for obj in related_objects if obj in current_detections)
                
        if context_count >= 2:
            boost += 0.03 * context_count
            
        return min(boost, 0.15)  # Giới hạn boost tối đa 15%
    
    def is_stable_detection(self, class_name, confidence, bbox_area):
        """Kiểm tra xem nhận diện có ổn định không (lọc nhiễu thông minh)"""
        # Thêm vào lịch sử với thông tin bổ sung
        detection_info = {
            'confidence': confidence,
            'area': bbox_area,
            'timestamp': time.time()
        }
        
        self.detection_history[class_name].append(detection_info)
        
        # Giữ chỉ max_history frame gần nhất
        if len(self.detection_history[class_name]) > self.max_history:
            self.detection_history[class_name].pop(0)
        
        # Cần ít nhất 4 frame để đánh giá
        if len(self.detection_history[class_name]) < 4:
            return False
        
        recent_detections = self.detection_history[class_name][-4:]
        threshold = self.get_confidence_threshold(class_name)
        
        # Áp dụng phạt nếu đối tượng này hay sai
        adjusted_threshold = threshold + self.false_positive_penalty.get(class_name, 0.0)
        
        # Kiểm tra độ ổn định về confidence
        confidence_stable = sum(1 for d in recent_detections if d['confidence'] > adjusted_threshold) >= 3
        
        # Kiểm tra độ ổn định về kích thước bounding box
        areas = [d['area'] for d in recent_detections]
        area_variance = np.var(areas) if len(areas) > 1 else 0
        area_stable = area_variance < (bbox_area * 0.3)  # Biến thiên kích thước < 30%
        
        # Kiểm tra tần suất xuất hiện
        time_gaps = []
        for i in range(1, len(recent_detections)):
            gap = recent_detections[i]['timestamp'] - recent_detections[i-1]['timestamp']
            time_gaps.append(gap)
        
        # Đối tượng thật thường xuất hiện liên tục
        frequency_stable = len(time_gaps) == 0 or max(time_gaps) < 2.0  # Không bị gián đoạn > 2s
        
        is_stable = confidence_stable and area_stable and frequency_stable
        
        # Cập nhật độ ổn định
        if is_stable:
            self.object_stability[class_name] += 1
            # Giảm penalty nếu ổn định
            if self.false_positive_penalty.get(class_name, 0) > 0:
                self.false_positive_penalty[class_name] *= 0.9
        else:
            # Tăng penalty nếu không ổn định (có thể là false positive)
            self.false_positive_penalty[class_name] = self.false_positive_penalty.get(class_name, 0) + 0.05
            
        return is_stable
    
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
    
    def draw_predictions(self, frame, results, all_detections=None):
        """Vẽ bounding box và label lên frame"""
        current_objects = Counter()
        if all_detections is None:
            all_detections = []
        
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
                        
                        # Tính toán thông tin bounding box
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1
                        bbox_area = bbox_width * bbox_height
                        
                        # Kiểm tra đa tầng thông minh
                        size_ok = self.is_reasonable_size(class_name, bbox_width, bbox_height, 
                                                        frame.shape[1], frame.shape[0])
                        position_ok = self.is_reasonable_position(class_name, x1, y1, x2, y2, 
                                                                frame.shape[1], frame.shape[0])
                        
                        # Áp dụng context boost
                        context_boost = self.get_context_boost(class_name, all_detections)
                        adjusted_confidence = confidence + context_boost
                        
                        # Kiểm tra confidence và độ ổn định
                        if (adjusted_confidence > confidence_threshold and 
                            size_ok and position_ok and
                            self.is_stable_detection(class_name, adjusted_confidence, bbox_area)):
                            # Nhận diện chính thức - màu xanh
                            current_objects[vietnamese_name] += 1
                            box_color = (0, 255, 0)  # Xanh lá
                            text_bg_color = (0, 255, 0)
                            status = "✓"
                            
                            # Thưởng cho detection đúng
                            if self.false_positive_penalty.get(class_name, 0) > 0:
                                self.false_positive_penalty[class_name] *= 0.8
                                
                        elif confidence > 0.5 and size_ok:
                            # Nghi ngờ - màu vàng
                            box_color = (0, 255, 255)  # Vàng
                            text_bg_color = (0, 255, 255)
                            
                            # Phân loại lý do nghi ngờ
                            if not position_ok:
                                status = "Pos?"
                            elif confidence <= confidence_threshold:
                                status = "Conf?"
                            else:
                                status = "Wait"
                        elif confidence > 0.3:
                            # Confidence thấp - màu đỏ nhạt
                            box_color = (128, 128, 255)  # Đỏ nhạt
                            text_bg_color = (128, 128, 255)
                            status = "Low"
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
                
                # Lấy danh sách tất cả detections để phân tích context
                all_detections = []
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            class_name = self.model.names[class_id]
                            confidence = box.conf[0]
                            if confidence > 0.3:  # Chỉ xét các detection có confidence tối thiểu
                                all_detections.append(class_name)
                
                # Vẽ predictions với thông tin context
                frame, current_objects = self.draw_predictions(frame, results, all_detections)
                
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
                
                # Thông tin chế độ lọc và thống kê
                cv2.putText(frame, "Xanh: OK | Vang: Nghi ngo | Do: Yeu", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Hiển thị số lượng đối tượng đang theo dõi
                tracking_count = len([k for k, v in self.object_stability.items() if v > 5])
                cv2.putText(frame, f"Dang theo doi: {tracking_count} doi tuong", 
                           (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                # Thêm hướng dẫn sử dụng (dùng ký tự an toàn)
                cv2.putText(frame, "Q: Thoat | S: Chup | R: Reset | L: Log | M: Mode", 
                           (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, f"Che do hien tai: {self.smart_mode}", 
                           (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
                
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
                elif key == ord('m'):  # Chuyển đổi chế độ thông minh
                    modes = ["normal", "strict", "sensitive"]
                    current_idx = modes.index(self.smart_mode)
                    self.smart_mode = modes[(current_idx + 1) % len(modes)]
                    print(f"🧠 Chuyển sang chế độ: {self.smart_mode}")
                    
                    # Điều chỉnh threshold theo chế độ
                    if self.smart_mode == "strict":
                        # Chế độ nghiêm ngặt - ít false positive
                        for key in self.confidence_thresholds:
                            if key != 'default':
                                self.confidence_thresholds[key] *= 1.2
                    elif self.smart_mode == "sensitive":
                        # Chế độ nhạy cảm - nhiều detection hơn
                        for key in self.confidence_thresholds:
                            if key != 'default':
                                self.confidence_thresholds[key] *= 0.8
        
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