"""
Component Detection Framework using YOLOv8
==========================================
Nâng cấp từ bài toán phân loại linh kiện lên nhận diện linh kiện trên bo mạch

Kiến trúc: YOLOv8 (Ultralytics)
- Backbone: CSPDarknet
- Neck: PANet (Path Aggregation Network)
- Head: Decoupled detection head

Hỗ trợ:
- Training từ scratch hoặc transfer learning
- Real-time detection từ webcam
- Inference trên images/videos
"""

import os
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import pandas as pd


class ComponentDetector:
    """
    Component Detection Model using YOLOv8
    
    Attributes:
        model_type: Loại model YOLOv8 ('n', 's', 'm', 'l', 'x')
        model: YOLO model instance
        class_names: List tên các classes
        colors: Dict màu sắc cho mỗi class
    """
    
    def __init__(self, model_type: str = 'n', pretrained: bool = True):
        """
        Initialize Component Detector
        
        Args:
            model_type: YOLOv8 model size
                - 'n': nano (fastest, least accurate)
                - 's': small
                - 'm': medium
                - 'l': large
                - 'x': xlarge (slowest, most accurate)
            pretrained: Sử dụng pretrained weights (COCO) hay không
        """
        self.model_type = model_type
        
        # Load model
        if pretrained:
            model_name = f'yolov8{model_type}.pt'
            print(f"Loading pretrained YOLOv8{model_type} model...")
        else:
            model_name = f'yolov8{model_type}.yaml'
            print(f"Initializing YOLOv8{model_type} from scratch...")
            
        self.model = YOLO(model_name)
        
        # Class names (sẽ được update sau khi load data.yaml)
        self.class_names = []
        self.colors = {}
        
        print(f"✓ Model initialized: YOLOv8{model_type}")
    
    def load_data_config(self, data_yaml_path: str):
        """
        Load data configuration từ data.yaml
        
        Args:
            data_yaml_path: Path to data.yaml file
        """
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        self.class_names = data_config['names']
        self.num_classes = data_config['nc']
        
        # Generate colors cho mỗi class
        self._generate_colors()
        
        print(f"✓ Loaded {self.num_classes} classes: {', '.join(self.class_names)}")
        
        return data_config
    
    def _generate_colors(self):
        """Generate unique colors cho mỗi class"""
        np.random.seed(42)
        for class_name in self.class_names:
            self.colors[class_name] = tuple(map(int, np.random.randint(50, 255, 3)))
    
    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        device = 0,  # Can be int (0, 1, etc.) or str ('cpu', '0', '1', etc.)
        project: str = 'runs/detect',
        name: str = 'component_detector',
        patience: int = 50,
        save_period: int = 10,
        **kwargs
    ):
        """
        Train the component detector
        
        Args:
            data_yaml: Path to data.yaml file
            epochs: Number of training epochs
            imgsz: Input image size
            batch: Batch size (-1 for auto)
            device: Device to train on (int: 0, 1, etc. for GPU or str: 'cpu' for CPU)
            project: Project directory
            name: Experiment name
            patience: Early stopping patience
            save_period: Save checkpoint every N epochs
            **kwargs: Additional training arguments
        
        Returns:
            Training results
        """
        print("\n" + "="*70)
        print("TRAINING COMPONENT DETECTOR")
        print("="*70)
        
        # Load data config
        self.load_data_config(data_yaml)
        
        # Training arguments
        train_args = {
            'data': data_yaml,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'device': device,
            'project': project,
            'name': name,
            'patience': patience,
            'save_period': save_period,
            'plots': True,
            'verbose': True,
            **kwargs
        }
        
        print(f"\nTraining Configuration:")
        print(f"  Model: YOLOv8{self.model_type}")
        print(f"  Classes: {self.num_classes}")
        print(f"  Epochs: {epochs}")
        print(f"  Image size: {imgsz}")
        print(f"  Batch size: {batch}")
        print(f"  Device: {device}")
        print(f"  Save to: {project}/{name}")
        print("\n" + "="*70 + "\n")
        
        # Train
        results = self.model.train(**train_args)
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED!")
        print("="*70)
        
        return results
    
    def validate(self, data_yaml: str = None, **kwargs):
        """
        Validate the model
        
        Args:
            data_yaml: Path to data.yaml (optional if already trained)
            **kwargs: Additional validation arguments
        
        Returns:
            Validation results
        """
        if data_yaml:
            val_args = {'data': data_yaml, **kwargs}
        else:
            val_args = kwargs
            
        results = self.model.val(**val_args)
        
        return results
    
    def predict(
        self,
        source,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        save: bool = False,
        save_txt: bool = False,
        save_conf: bool = False,
        **kwargs
    ):
        """
        Run inference
        
        Args:
            source: Image path, directory, video, or webcam (0)
            conf: Confidence threshold
            iou: NMS IoU threshold
            imgsz: Input image size
            save: Save results
            save_txt: Save results as .txt
            save_conf: Save confidence in .txt
            **kwargs: Additional prediction arguments
        
        Returns:
            Prediction results
        """
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            save=save,
            save_txt=save_txt,
            save_conf=save_conf,
            **kwargs
        )
        
        return results
    
    def export(self, format: str = 'onnx', **kwargs):
        """
        Export model to different formats
        
        Args:
            format: Export format ('onnx', 'torchscript', 'coreml', 'tflite', etc.)
            **kwargs: Additional export arguments
        
        Returns:
            Export path
        """
        export_path = self.model.export(format=format, **kwargs)
        print(f"✓ Model exported to: {export_path}")
        
        return export_path
    
    def load_weights(self, weights_path: str):
        """
        Load trained weights
        
        Args:
            weights_path: Path to weights file (.pt)
        """
        self.model = YOLO(weights_path)
        print(f"✓ Loaded weights from: {weights_path}")
    
    def visualize_predictions(
        self,
        image_path: str,
        conf: float = 0.25,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Visualize predictions on a single image
        
        Args:
            image_path: Path to image
            conf: Confidence threshold
            save_path: Path to save visualization
            show: Show the plot
        """
        # Run prediction
        results = self.predict(image_path, conf=conf, save=False)
        
        # Load image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img)
        
        # Draw predictions
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf_score = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.class_names[class_id]
                
                # Draw rectangle
                width = x2 - x1
                height = y2 - y1
                
                color = np.array(self.colors[class_name]) / 255.0
                
                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=2,
                    edgecolor=color,
                    facecolor='none'
                )
                ax.add_patch(rect)
                
                # Draw label
                label = f"{class_name}: {conf_score:.2f}"
                ax.text(
                    x1, y1 - 5,
                    label,
                    color='white',
                    fontsize=10,
                    bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=2)
                )
        
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"✓ Visualization saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


class WebcamDetector:
    """
    Real-time component detection từ webcam
    """
    
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        """
        Initialize webcam detector
        
        Args:
            model_path: Path to trained model (.pt file)
            conf_threshold: Confidence threshold
            iou_threshold: NMS IoU threshold
        """
        print(f"Loading model from {model_path}...")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Get class names
        self.class_names = self.model.names
        
        # Generate colors
        self.colors = self._generate_colors()
        
        print("✓ Model loaded successfully!")
        print(f"✓ Classes: {len(self.class_names)}")
    
    def _generate_colors(self):
        """Generate unique colors cho mỗi class"""
        np.random.seed(42)
        colors = {}
        for class_id, class_name in self.class_names.items():
            colors[class_id] = tuple(map(int, np.random.randint(50, 255, 3)))
        return colors
    
    def run(
        self,
        camera_id: int = 0,
        window_name: str = "Component Detector",
        display_fps: bool = True
    ):
        """
        Run real-time detection từ webcam
        
        Args:
            camera_id: Camera ID (0 for default webcam)
            window_name: Window name
            display_fps: Display FPS on frame
        """
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Cannot open camera {camera_id}")
            return
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print("\n" + "="*70)
        print("REAL-TIME COMPONENT DETECTION")
        print("="*70)
        print(f"Camera ID: {camera_id}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Confidence threshold: {self.conf_threshold}")
        print(f"IoU threshold: {self.iou_threshold}")
        print(f"\nControls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save current frame")
        print("  - Press 'p' to pause/resume")
        print("  - Press '+' to increase confidence threshold")
        print("  - Press '-' to decrease confidence threshold")
        print("="*70 + "\n")
        
        # FPS calculation
        fps_start_time = datetime.now()
        fps_frame_count = 0
        current_fps = 0
        
        paused = False
        frame_count = 0
        
        try:
            while True:
                if not paused:
                    # Read frame
                    ret, frame = cap.read()
                    
                    if not ret:
                        print("Error: Cannot read frame")
                        break
                    
                    frame_count += 1
                    
                    # Run detection
                    results = self.model.predict(
                        frame,
                        conf=self.conf_threshold,
                        iou=self.iou_threshold,
                        verbose=False
                    )
                    
                    # Draw detections
                    detection_count = 0
                    for result in results:
                        boxes = result.boxes
                        
                        for box in boxes:
                            detection_count += 1
                            
                            # Get box info
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            conf = float(box.conf[0].cpu().numpy())
                            class_id = int(box.cls[0].cpu().numpy())
                            class_name = self.class_names[class_id]
                            
                            # Get color
                            color = self.colors[class_id]
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Prepare label
                            label = f"{class_name}: {conf:.2f}"
                            
                            # Get label size
                            (label_w, label_h), baseline = cv2.getTextSize(
                                label,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                1
                            )
                            
                            # Draw label background
                            cv2.rectangle(
                                frame,
                                (x1, y1 - label_h - baseline - 5),
                                (x1 + label_w, y1),
                                color,
                                -1
                            )
                            
                            # Draw label text
                            cv2.putText(
                                frame,
                                label,
                                (x1, y1 - baseline - 2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA
                            )
                    
                    # Calculate FPS
                    fps_frame_count += 1
                    if fps_frame_count >= 10:
                        fps_end_time = datetime.now()
                        current_fps = fps_frame_count / (fps_end_time - fps_start_time).total_seconds()
                        fps_start_time = fps_end_time
                        fps_frame_count = 0
                    
                    # Draw info overlay
                    if display_fps:
                        info_text = [
                            f"FPS: {current_fps:.1f}",
                            f"Detections: {detection_count}",
                            f"Conf: {self.conf_threshold:.2f}",
                            f"Frame: {frame_count}"
                        ]
                        
                        y_offset = 30
                        for text in info_text:
                            # Draw text background
                            (text_w, text_h), _ = cv2.getTextSize(
                                text,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                2
                            )
                            cv2.rectangle(
                                frame,
                                (5, y_offset - text_h - 5),
                                (15 + text_w, y_offset + 5),
                                (0, 0, 0),
                                -1
                            )
                            
                            # Draw text
                            cv2.putText(
                                frame,
                                text,
                                (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2,
                                cv2.LINE_AA
                            )
                            y_offset += 30
                
                # Show frame
                cv2.imshow(window_name, frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"detection_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"✓ Frame saved: {filename}")
                elif key == ord('p'):
                    # Pause/Resume
                    paused = not paused
                    status = "PAUSED" if paused else "RESUMED"
                    print(f"\n{status}")
                elif key == ord('+') or key == ord('='):
                    # Increase confidence threshold
                    self.conf_threshold = min(0.95, self.conf_threshold + 0.05)
                    print(f"\nConfidence threshold: {self.conf_threshold:.2f}")
                elif key == ord('-') or key == ord('_'):
                    # Decrease confidence threshold
                    self.conf_threshold = max(0.05, self.conf_threshold - 0.05)
                    print(f"\nConfidence threshold: {self.conf_threshold:.2f}")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            print("\n" + "="*70)
            print(f"Total frames processed: {frame_count}")
            print(f"Average FPS: {current_fps:.1f}")
            print("Camera closed.")
            print("="*70 + "\n")


def plot_training_results(results_dir: str):
    """
    Plot training results từ CSV files
    
    Args:
        results_dir: Directory chứa results.csv
    """
    csv_path = os.path.join(results_dir, 'results.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return
    
    # Read results
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # Remove whitespace
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Loss curves
    axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Box Loss', marker='o')
    axes[0, 0].plot(df['epoch'], df['train/cls_loss'], label='Class Loss', marker='s')
    axes[0, 0].plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss', marker='^')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: mAP
    if 'metrics/mAP50(B)' in df.columns:
        axes[0, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', marker='o')
        axes[0, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', marker='s')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mAP')
        axes[0, 1].set_title('Validation mAP')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Precision & Recall
    if 'metrics/precision(B)' in df.columns:
        axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision', marker='o')
        axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall', marker='s')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Precision & Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Learning Rate
    if 'lr/pg0' in df.columns:
        axes[1, 1].plot(df['epoch'], df['lr/pg0'], label='LR pg0', marker='o')
        axes[1, 1].plot(df['epoch'], df['lr/pg1'], label='LR pg1', marker='s')
        axes[1, 1].plot(df['epoch'], df['lr/pg2'], label='LR pg2', marker='^')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(results_dir, 'training_analysis.png')
    plt.savefig(save_path, dpi=150)
    print(f"✓ Training analysis saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("Component Detector Module")
    print("=" * 70)
    print("This module provides:")
    print("  - ComponentDetector: Main detection model class")
    print("  - WebcamDetector: Real-time webcam detection")
    print("  - plot_training_results: Visualize training metrics")
    print("\nUsage examples:")
    print("  1. Training: See train_detector.py")
    print("  2. Webcam: See webcam_detector.py")
    print("=" * 70)
