# PowerShell script to run Object Detection App
# Author: AI Assistant
# Date: 28/09/2025

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   OBJECT DETECTION APP LAUNCHER" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
$pythonVersion = python --version 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "🐍 Python version: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "❌ Python not found! Please install Python first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if requirements are installed
Write-Host "🔍 Checking requirements..." -ForegroundColor Yellow

$requiredPackages = @("ultralytics", "opencv", "numpy", "pandas")
$missingPackages = @()

foreach ($package in $requiredPackages) {
    python -c "import $package" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ $package is installed" -ForegroundColor Green
    } else {
        Write-Host "❌ $package is missing" -ForegroundColor Red
        $missingPackages += $package
    }
}

if ($missingPackages.Count -gt 0) {
    Write-Host "⚠️  Missing packages detected!" -ForegroundColor Yellow
    Write-Host "Missing: $($missingPackages -join ', ')" -ForegroundColor Red
    
    $install = Read-Host "Do you want to install missing packages? (y/n)"
    if ($install.ToLower() -eq 'y') {
        Write-Host "📦 Installing requirements..." -ForegroundColor Yellow
        pip install -r requirements.txt
    } else {
        Write-Host "❌ Cannot run app without required packages!" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Run system check
Write-Host ""
Write-Host "🔍 Running system check..." -ForegroundColor Yellow
python check_system.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  System check failed. Please fix issues before running the app." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Ask user what to do
Write-Host ""
Write-Host "📋 Choose an option:" -ForegroundColor Cyan
Write-Host "1. Run Object Detection App" -ForegroundColor White
Write-Host "2. Test YOLOv8 Model" -ForegroundColor White
Write-Host "3. Exit" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Enter your choice (1-3)"

switch ($choice) {
    "1" {
        Write-Host "🚀 Starting Object Detection App..." -ForegroundColor Green
        python object_detection_app.py
    }
    "2" {
        Write-Host "🧪 Starting YOLOv8 Test..." -ForegroundColor Green
        python test_yolo.py
    }
    "3" {
        Write-Host "👋 Goodbye!" -ForegroundColor Yellow
        exit 0
    }
    default {
        Write-Host "❌ Invalid choice!" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "✅ Script completed!" -ForegroundColor Green
Read-Host "Press Enter to exit"