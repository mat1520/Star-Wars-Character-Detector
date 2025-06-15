import os
import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Union, Tuple
import ultralytics
from ultralytics.nn.modules.block import Bottleneck
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f, SPPF
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel
from torch.nn import Sequential
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.activation import SiLU
from torch.nn.modules.container import ModuleList
from torch.nn.modules.pooling import MaxPool2d
from torch.serialization import add_safe_globals
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.upsampling import Upsample

# Add safe globals for model loading
add_safe_globals([
    Upsample,
    MaxPool2d,
    Bottleneck,
    ModuleList,
    DetectionModel,
    Sequential,
    Conv,
    C2f,
    SPPF,
    Detect,
    Conv2d,
    BatchNorm2d,
    SiLU
])

class StarWarsDetector:
    def __init__(self, model_path: Union[str, Path] = "runs/detect/star_wars_detector/weights/best.pt"):
        """
        Initialize the Star Wars character detector.
        
        Args:
            model_path (str or Path): Path to the trained YOLOv8 model
        """
        # Cargar el modelo con weights_only=False para compatibilidad
        self.model = YOLO(model_path, task='detect')
        self.model.model.load_state_dict(torch.load(model_path, weights_only=False)['model'].state_dict())
        self.class_names = [
            "Darth Vader",
            "Luke Skywalker",
            "Yoda",
            "R2-D2",
            "C-3PO",
            "Chewbacca",
            "Han Solo",
            "Leia Organa"
        ]
    
    def detect_characters(
        self,
        image: Union[str, Path, np.ndarray],
        conf_threshold: float = 0.25,
        return_image: bool = True
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Detect Star Wars characters in an image.
        
        Args:
            image: Path to image or numpy array
            conf_threshold: Confidence threshold for detections
            return_image: Whether to return the image with bounding boxes
            
        Returns:
            Tuple containing:
            - List of detections, each as a dict with keys:
              'box': [x1, y1, x2, y2]
              'label': character name
              'confidence': confidence score
            - Image with bounding boxes (if return_image=True)
        """
        # Read image if path is provided
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"Could not read image from {image}")
        
        # Make prediction
        results = self.model(image, conf=conf_threshold)[0]
        
        # Process detections
        detections = []
        for box in results.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get class and confidence
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # Create detection dict
            detection = {
                'box': [x1, y1, x2, y2],
                'label': self.class_names[class_id],
                'confidence': confidence
            }
            detections.append(detection)
            
            # Draw bounding box if return_image is True
            if return_image:
                # Draw box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{detection['label']}: {confidence:.2f}"
                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
        
        return detections, image if return_image else None

def main():
    """Example usage of the StarWarsDetector class."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect Star Wars characters in an image")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--model", default="runs/detect/star_wars_detector/weights/best.pt",
                      help="Path to the trained model")
    parser.add_argument("--conf", type=float, default=0.25,
                      help="Confidence threshold")
    parser.add_argument("--output", help="Path to save the output image")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = StarWarsDetector(args.model)
    
    # Detect characters
    detections, image = detector.detect_characters(args.image_path, args.conf)
    
    # Print detections
    print("\nDetections:")
    for det in detections:
        print(f"{det['label']}: {det['confidence']:.2f}")
    
    # Save output image if requested
    if args.output and image is not None:
        cv2.imwrite(args.output, image)
        print(f"\nOutput image saved to: {args.output}")

if __name__ == "__main__":
    main() 