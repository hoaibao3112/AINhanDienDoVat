#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script kiểm tra hệ thống và camera cho ứng dụng Object Detection
"""

import cv2
import platform
import sys
import torch

def check_python_version():
    """Kiểm tra phiên bản Python"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major >= 3 and version.minor >= 8:
        print("✅ Python version hỗ trợ")
        return True
    else:
        print("❌ Cần Python 3.8 trở lên")
        return False

def check_camera():
    """Kiểm tra camera"""
    print("\n📹 Kiểm tra camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Không thể mở camera")
        return False
    
    # Thử đọc frame
    ret, frame = cap.read()
    if ret:
        height, width = frame.shape[:2]
        print(f"✅ Camera hoạt động - Độ phân giải: {width}x{height}")
        cap.release()
        return True
    else:
        print("❌ Không thể đọc dữ liệu từ camera")
        cap.release()
        return False

def check_pytorch():
    """Kiểm tra PyTorch và CUDA"""
    print("\n🔥 Kiểm tra PyTorch...")
    try:
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA khả dụng - GPU: {torch.cuda.get_device_name()}")
            return True
        else:
            print("⚠️  CUDA không khả dụng - sẽ sử dụng CPU")
            return True
    except ImportError:
        print("❌ PyTorch chưa được cài đặt")
        return False

def check_opencv():
    """Kiểm tra OpenCV"""
    print("\n📸 Kiểm tra OpenCV...")
    try:
        print(f"OpenCV version: {cv2.__version__}")
        print("✅ OpenCV đã sẵn sàng")
        return True
    except ImportError:
        print("❌ OpenCV chưa được cài đặt")
        return False

def main():
    print("=" * 50)
    print("🔍 KIỂM TRA HỆ THỐNG CHO OBJECT DETECTION APP")
    print("=" * 50)
    
    print(f"\n💻 Hệ điều hành: {platform.system()} {platform.release()}")
    
    checks = [
        check_python_version(),
        check_opencv(),
        check_pytorch(),
        check_camera()
    ]
    
    print("\n" + "=" * 50)
    if all(checks):
        print("🎉 Hệ thống sẵn sàng cho Object Detection!")
    else:
        print("⚠️  Cần khắc phục một số vấn đề trước khi chạy ứng dụng")
    print("=" * 50)

if __name__ == "__main__":
    main()