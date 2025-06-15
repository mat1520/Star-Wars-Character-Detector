import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import shutil

class AutoAnnotator:
    def __init__(self, dataset_dir="dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.train_dir = self.dataset_dir / "train"
        self.val_dir = self.dataset_dir / "val"
        
        # Initialize HOG person detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Character classes
        self.classes = [
            "Darth Vader",
            "Luke Skywalker",
            "Yoda",
            "R2-D2",
            "C-3PO",
            "Chewbacca",
            "Han Solo",
            "Leia Organa"
        ]
        
        # Create backup of original labels
        self._backup_labels()
        
    def _backup_labels(self):
        """Create backup of original labels."""
        backup_dir = self.dataset_dir / "labels_backup"
        if not backup_dir.exists():
            shutil.copytree(self.train_dir / "labels", backup_dir / "train")
            shutil.copytree(self.val_dir / "labels", backup_dir / "val")
            print("Created backup of original labels")
    
    def annotate_images(self, conf_threshold=0.5):
        """Annotate images using HOG person detector."""
        print("Starting automatic annotation...")
        
        # Process training set
        print("\nProcessing training set...")
        self._process_directory(self.train_dir, conf_threshold)
        
        # Process validation set
        print("\nProcessing validation set...")
        self._process_directory(self.val_dir, conf_threshold)
        
        print("\nAnnotation complete!")
    
    def _process_directory(self, directory, conf_threshold):
        """Process all images in a directory."""
        images_dir = directory / "images"
        labels_dir = directory / "labels"
        
        # Get all images
        image_files = list(images_dir.glob("*.jpg"))
        
        for img_path in tqdm(image_files, desc="Annotating images"):
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Could not read image: {img_path}")
                continue
            
            # Resize image if too large
            max_dimension = 1000
            height, width = img.shape[:2]
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                img = cv2.resize(img, None, fx=scale, fy=scale)
            
            # Detect people
            boxes, weights = self.hog.detectMultiScale(
                img,
                winStride=(8, 8),
                padding=(4, 4),
                scale=1.05,
                hitThreshold=0
            )
            
            # Create label file
            label_path = labels_dir / f"{img_path.stem}.txt"
            with open(label_path, "w") as f:
                # Process each detection
                for box, weight in zip(boxes, weights):
                    if weight >= conf_threshold:
                        # Get box coordinates
                        x, y, w, h = box
                        img_height, img_width = img.shape[:2]
                        
                        # Convert to YOLO format (x_center, y_center, width, height)
                        x_center = (x + w/2) / img_width
                        y_center = (y + h/2) / img_height
                        width = w / img_width
                        height = h / img_height
                        
                        # Write to file (using class 0 for person)
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def verify_annotations(self):
        """Verify that all images have corresponding label files."""
        print("\nVerifying annotations...")
        
        # Check training set
        train_images = set(f.stem for f in (self.train_dir / "images").glob("*.jpg"))
        train_labels = set(f.stem for f in (self.train_dir / "labels").glob("*.txt"))
        missing_train = train_images - train_labels
        
        # Check validation set
        val_images = set(f.stem for f in (self.val_dir / "images").glob("*.jpg"))
        val_labels = set(f.stem for f in (self.val_dir / "labels").glob("*.txt"))
        missing_val = val_images - val_labels
        
        if missing_train or missing_val:
            print("\nWarning: Some images are missing annotations:")
            if missing_train:
                print(f"\nTraining set: {len(missing_train)} images missing annotations")
            if missing_val:
                print(f"Validation set: {len(missing_val)} images missing annotations")
        else:
            print("All images have corresponding annotations!")
        
        # Print statistics
        print(f"\nTotal training images: {len(train_images)}")
        print(f"Total validation images: {len(val_images)}")
        print(f"Total training annotations: {len(train_labels)}")
        print(f"Total validation annotations: {len(val_labels)}")

def main():
    # Create annotator
    annotator = AutoAnnotator()
    
    # Run annotation
    annotator.annotate_images(conf_threshold=0.5)
    
    # Verify results
    annotator.verify_annotations()

if __name__ == "__main__":
    main() 