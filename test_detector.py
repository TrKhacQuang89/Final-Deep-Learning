"""
Test Detector on Images
=======================
Script để test trained model trên test images

Usage:
    python test_detector.py --weights runs/detect/component_detector/weights/best.pt --source test/images
"""

import argparse
from pathlib import Path
from component_detector import ComponentDetector
import os


def main():
    parser = argparse.ArgumentParser(description='Test Component Detector')
    
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to trained model weights (.pt file)'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Path to test images directory or single image'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data.yaml',
        help='Path to data.yaml file'
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
        '--imgsz',
        type=int,
        default=640,
        help='Input image size'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save detection results'
    )
    parser.add_argument(
        '--save-txt',
        action='store_true',
        help='Save results as .txt files'
    )
    parser.add_argument(
        '--save-conf',
        action='store_true',
        help='Save confidence scores in .txt files'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='runs/detect',
        help='Project directory'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='test',
        help='Experiment name'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize predictions with matplotlib'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='Device (0 for GPU, cpu for CPU)'
    )
    
    args = parser.parse_args()
    
    # Check if weights exist
    if not Path(args.weights).exists():
        print(f"Error: Weights file not found: {args.weights}")
        return
    
    # Check if source exists
    if not Path(args.source).exists():
        print(f"Error: Source not found: {args.source}")
        return
    
    print("\n" + "="*70)
    print("COMPONENT DETECTOR TESTING")
    print("="*70)
    print(f"Weights: {args.weights}")
    print(f"Source: {args.source}")
    print(f"Confidence: {args.conf}")
    print(f"IoU: {args.iou}")
    print("="*70 + "\n")
    
    # Initialize detector
    detector = ComponentDetector(model_type='n', pretrained=False)
    detector.load_weights(args.weights)
    
    # Load data config
    if Path(args.data).exists():
        detector.load_data_config(args.data)
    
    # Run prediction
    print("Running detection...")
    results = detector.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        save=args.save,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        project=args.project,
        name=args.name,
        device=args.device
    )
    
    # Print results
    print("\n" + "="*70)
    print("DETECTION RESULTS")
    print("="*70)
    
    for i, result in enumerate(results):
        print(f"\nImage {i+1}: {result.path}")
        print(f"  Detections: {len(result.boxes)}")
        
        # Count detections per class
        if len(result.boxes) > 0:
            class_counts = {}
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = detector.class_names[class_id]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            print("  Class distribution:")
            for class_name, count in sorted(class_counts.items()):
                print(f"    - {class_name}: {count}")
    
    # Visualize if requested
    if args.visualize and Path(args.source).is_file():
        print("\n" + "="*70)
        print("VISUALIZATION")
        print("="*70)
        
        save_path = Path(args.project) / args.name / "visualization.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        detector.visualize_predictions(
            image_path=args.source,
            conf=args.conf,
            save_path=str(save_path),
            show=False
        )
    
    print("\n" + "="*70)
    print("TESTING COMPLETED")
    print("="*70)
    if args.save:
        print(f"Results saved to: {Path(args.project) / args.name}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
