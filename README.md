# Component Detection Project - Final Project

## Giới thiệu

Dự án này nâng cấp từ bài toán **phân loại linh kiện** (classification) lên **nhận diện linh kiện trên bo mạch** (object detection). Sử dụng YOLOv8 để detect và phân loại nhiều linh kiện điện tử trong một ảnh bo mạch.

### Nâng cấp từ Midterm Project

**Midterm (Classification):**
- Input: Ảnh chứa 1 linh kiện duy nhất
- Output: Class của linh kiện đó
- Model: MLP Network (NumPy)

**Final (Object Detection):**
- Input: Ảnh bo mạch chứa nhiều linh kiện
- Output: Bounding boxes + classes của tất cả linh kiện
- Model: YOLOv8 (Ultralytics)
- Bonus: Real-time detection từ webcam

## Dataset

Dataset gồm 10 classes linh kiện điện tử:
1. Capacitor
2. Ceramic Capacitor
3. Diode
4. IC
5. LED
6. Potentiometer
7. Resistor
8. Transformer
9. Trigger Button
10. Voltage Regulator

**Cấu trúc:**
```
learn_final/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

## Cài đặt

### 1. Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**Lưu ý:** Nếu có GPU NVIDIA, cài đặt PyTorch với CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Sử dụng

### 1. Training

Train model với YOLOv8 nano (nhanh nhất):

```bash
python train_detector.py --model n --epochs 100 --batch 16
```

Train với YOLOv8 small (cân bằng tốc độ/độ chính xác):

```bash
python train_detector.py --model s --epochs 100 --batch 16
```

Train với YOLOv8 medium (độ chính xác cao hơn):

```bash
python train_detector.py --model m --epochs 150 --batch 8
```

**Các tham số quan trọng:**
- `--model`: Kích thước model (n/s/m/l/x)
- `--epochs`: Số epochs
- `--batch`: Batch size
- `--imgsz`: Kích thước ảnh input (default: 640)
- `--device`: Device (0 cho GPU, cpu cho CPU)
- `--patience`: Early stopping patience
- `--lr0`: Learning rate ban đầu

**Ví dụ training đầy đủ:**

```bash
python train_detector.py \
    --model s \
    --epochs 150 \
    --batch 16 \
    --imgsz 640 \
    --device 0 \
    --patience 50 \
    --lr0 0.01 \
    --save-period 10
```

### 2. Testing

Test model trên test set:

```bash
python test_detector.py \
    --weights runs/detect/component_detector/weights/best.pt \
    --source test/images \
    --conf 0.25 \
    --save
```

Test trên một ảnh cụ thể với visualization:

```bash
python test_detector.py \
    --weights runs/detect/component_detector/weights/best.pt \
    --source test/images/sample.jpg \
    --conf 0.25 \
    --visualize
```

### 3. Real-time Webcam Detection

Chạy detection real-time từ webcam:

```bash
python webcam_detector.py \
    --weights runs/detect/component_detector/weights/best.pt \
    --camera 0 \
    --conf 0.25
```

**Controls trong webcam mode:**
- `q`: Thoát
- `s`: Lưu frame hiện tại
- `p`: Pause/Resume
- `+`: Tăng confidence threshold
- `-`: Giảm confidence threshold

### 4. Sử dụng trong Python Code

```python
from component_detector import ComponentDetector, WebcamDetector

# Training
detector = ComponentDetector(model_type='n', pretrained=True)
detector.train(
    data_yaml='data.yaml',
    epochs=100,
    batch=16,
    device='0'
)

# Inference
detector.load_weights('runs/detect/component_detector/weights/best.pt')
results = detector.predict('test/images/sample.jpg', conf=0.25)

# Webcam
webcam = WebcamDetector(
    model_path='runs/detect/component_detector/weights/best.pt',
    conf_threshold=0.25
)
webcam.run(camera_id=0)
```

## Cấu trúc Project

```
learn_final/
├── component_detector.py      # Core detection module
├── train_detector.py           # Training script
├── test_detector.py            # Testing script
├── webcam_detector.py          # Real-time webcam detection
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── data.yaml                   # Dataset configuration
├── train/                      # Training data
├── valid/                      # Validation data
├── test/                       # Test data
└── runs/                       # Training results (auto-generated)
    └── detect/
        └── component_detector/
            ├── weights/
            │   ├── best.pt     # Best model weights
            │   └── last.pt     # Last epoch weights
            ├── results.csv     # Training metrics
            └── *.png           # Training plots
```

## Kết quả Training

Sau khi training, kết quả sẽ được lưu trong `runs/detect/component_detector/`:

- **weights/best.pt**: Model tốt nhất (theo validation mAP)
- **weights/last.pt**: Model ở epoch cuối cùng
- **results.csv**: Metrics theo từng epoch
- **confusion_matrix.png**: Confusion matrix
- **results.png**: Training curves (loss, mAP, precision, recall)
- **training_analysis.png**: Phân tích chi tiết (custom plot)

## Metrics Đánh giá

- **mAP@0.5**: Mean Average Precision tại IoU threshold 0.5
- **mAP@0.5:0.95**: Mean Average Precision trung bình từ IoU 0.5 đến 0.95
- **Precision**: Tỉ lệ detections đúng trong tất cả detections
- **Recall**: Tỉ lệ objects được detect trong tất cả ground truth objects
- **Box Loss**: Loss cho bounding box regression
- **Class Loss**: Loss cho classification
- **DFL Loss**: Distribution Focal Loss

## So sánh Model Sizes

| Model | Size | Speed | mAP | Use Case |
|-------|------|-------|-----|----------|
| YOLOv8n | 3.2M params | Fastest | Lowest | Real-time, embedded |
| YOLOv8s | 11.2M params | Fast | Medium | Balanced |
| YOLOv8m | 25.9M params | Medium | High | Accuracy priority |
| YOLOv8l | 43.7M params | Slow | Higher | High accuracy |
| YOLOv8x | 68.2M params | Slowest | Highest | Best accuracy |

**Khuyến nghị:**
- **Real-time webcam**: YOLOv8n hoặc YOLOv8s
- **Cân bằng**: YOLOv8s hoặc YOLOv8m
- **Độ chính xác cao**: YOLOv8m hoặc YOLOv8l

## Troubleshooting

### 1. CUDA Out of Memory

Giảm batch size:
```bash
python train_detector.py --model n --batch 8
```

Hoặc giảm image size:
```bash
python train_detector.py --model n --imgsz 416
```

### 2. Webcam không hoạt động

Thử camera ID khác:
```bash
python webcam_detector.py --weights best.pt --camera 1
```

Kiểm tra OpenCV:
```python
import cv2
cap = cv2.VideoCapture(0)
print(cap.isOpened())
```

### 3. Training quá chậm

- Sử dụng GPU: `--device 0`
- Giảm workers: `--workers 4`
- Sử dụng model nhỏ hơn: `--model n`

## Tips để cải thiện Performance

1. **Data Augmentation**: Điều chỉnh augmentation parameters
   ```bash
   python train_detector.py --model s --mosaic 1.0 --mixup 0.1 --fliplr 0.5
   ```

2. **Learning Rate**: Thử learning rate khác
   ```bash
   python train_detector.py --model s --lr0 0.001 --lrf 0.01
   ```

3. **Image Size**: Tăng image size (nếu có GPU mạnh)
   ```bash
   python train_detector.py --model s --imgsz 800
   ```

4. **Epochs**: Train lâu hơn với early stopping
   ```bash
   python train_detector.py --model s --epochs 300 --patience 100
   ```

## Tài liệu tham khảo

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
- [YOLO Paper](https://arxiv.org/abs/1506.02640)

## License

Dataset: CC BY 4.0 (Roboflow)
Code: MIT License

## Tác giả

Dự án cuối kỳ - Nâng cấp từ Classification lên Object Detection
