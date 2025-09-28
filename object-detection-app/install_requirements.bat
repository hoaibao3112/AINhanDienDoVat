@echo off
echo ========================================
echo   CAI DAT REQUIREMENTS CHO OBJECT DETECTION APP
echo ========================================
echo.

echo 🔄 Dang cap nhat pip...
python -m pip install --upgrade pip

echo.
echo 📦 Dang cai dat cac thu vien...
pip install -r requirements.txt

echo.
echo 🎯 Dang tai YOLOv8 model...
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('Model downloaded successfully!')"

echo.
echo ✅ CAI DAT HOAN TAT!
echo.
echo 📋 Huong dan su dung:
echo    1. Chay 'python check_system.py' de kiem tra he thong
echo    2. Chay 'python test_yolo.py' de test YOLOv8
echo    3. Chay 'python object_detection_app.py' de bat dau ung dung
echo.
pause