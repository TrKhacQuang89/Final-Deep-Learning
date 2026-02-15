"""
Real-time Component Detection from Webcam
=========================================
Ứng dụng real-time detection sử dụng trained YOLOv8 model

Usage:
    python webcam_detector.py --weights runs/detect/component_detector/weights/best.pt
"""

import argparse
from component_detector import WebcamDetector


def main():
    parser = argparse.ArgumentParser(description='Real-time Component Detection')
    
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to trained model weights (.pt file)'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera ID (0 for default webcam)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='NMS IoU threshold'
    )
    parser.add_argument(
        '--window-name',
        type=str,
        default='Component Detector',
        help='Window name'
    )
    parser.add_argument(
        '--no-fps',
        action='store_true',
        help='Disable FPS display'
    )
    
    args = parser.parse_args()
    
    # Initialize webcam detector
    detector = WebcamDetector(
        model_path=args.weights,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Run detection
    detector.run(
        camera_id=args.camera,
        window_name=args.window_name,
        display_fps=not args.no_fps
    )


if __name__ == "__main__":
    main()
