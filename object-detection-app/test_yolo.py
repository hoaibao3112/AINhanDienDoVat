#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script test YOLOv8 model
"""

import cv2
from ultralytics import YOLO
import time

def test_yolo_basic():
    """Test cơ bản YOLOv8"""
    print("🧪 Testing YOLOv8 Model...")
    
    try:
        # Tải model
        print("📦 Loading YOLOv8 model...")
        model = YOLO('yolov8n.pt')
        print("✅ Model loaded successfully!")
        
        # Hiển thị thông tin model
        print(f"📊 Model info:")
        print(f"   - Classes: {len(model.names)}")
        print(f"   - Model type: YOLOv8")
        
        # Test với camera
        print("📹 Testing with camera...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return False
        
        print("✅ Camera opened successfully!")
        print("Press 'q' to quit test")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Chỉ xử lý mỗi 5 frame để tăng tốc độ
            if frame_count % 5 == 0:
                results = model(frame, verbose=False)
                
                # Vẽ kết quả
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            confidence = box.conf[0]
                            class_id = int(box.cls[0])
                            class_name = model.names[class_id]
                            
                            if confidence > 0.5:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                label = f"{class_name}: {confidence:.2f}"
                                cv2.putText(frame, label, (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Hiển thị FPS
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('YOLOv8 Test - Press Q to quit', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"📊 Test completed!")
        print(f"   - Total frames: {frame_count}")
        print(f"   - Average FPS: {fps:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def list_yolo_classes():
    """Liệt kê các class mà YOLOv8 có thể nhận diện"""
    try:
        model = YOLO('yolov8n.pt')
        print("\n📋 YOLOv8 có thể nhận diện các đối tượng sau:")
        print("="*60)
        
        for i, class_name in model.names.items():
            print(f"{i:2d}: {class_name}")
        
        print("="*60)
        print(f"Tổng cộng: {len(model.names)} classes")
        
        # Các class phổ biến
        common_objects = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
                         'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
                         'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 
                         'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
                         'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
                         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
                         'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
                         'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 
                         'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 
                         'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
                         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
                         'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
                         'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
                         'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        
        print(f"\n🎯 Các đối tượng phổ biến trong nhà:")
        for obj in common_objects:
            if obj in model.names.values():
                print(f"   ✅ {obj}")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")

def main():
    """Hàm main"""
    print("🚀 YOLOv8 Test Suite")
    print("="*50)
    
    # Liệt kê classes
    list_yolo_classes()
    
    # Test model
    input("\n👆 Nhấn Enter để bắt đầu test camera...")
    test_yolo_basic()

if __name__ == "__main__":
    main()