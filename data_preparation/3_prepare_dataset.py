import os
import shutil
import random
from pathlib import Path
import yaml
from tqdm import tqdm

class DatasetPreparator:
    def __init__(self, raw_dir="dataset_raw", output_dir="dataset"):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "val"
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        
        # Create directories
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)
        (self.train_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.train_dir / "labels").mkdir(parents=True, exist_ok=True)
        (self.val_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.val_dir / "labels").mkdir(parents=True, exist_ok=True)
        
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
        
        # Create class mapping
        self.class_to_id = {name: idx for idx, name in enumerate(self.classes)}
        
    def split_dataset(self, val_split=0.2):
        """Split dataset into training and validation sets."""
        print("Splitting dataset into training and validation sets...")
        
        for character in self.classes:
            char_dir = self.raw_dir / character.replace(" ", "_")
            if not char_dir.exists():
                print(f"Warning: No images found for {character}")
                continue
                
            # Get all images for this character
            images = list(char_dir.glob("*.jpg"))
            random.shuffle(images)
            
            # Calculate split
            val_size = int(len(images) * val_split)
            train_images = images[val_size:]
            val_images = images[:val_size]
            
            # Copy images and create labels
            self._process_images(train_images, character, is_train=True)
            self._process_images(val_images, character, is_train=False)
            
    def _process_images(self, images, character, is_train=True):
        """Process images and create YOLO format labels."""
        target_dir = self.train_dir if is_train else self.val_dir
        class_id = self.class_to_id[character]
        
        for img_path in tqdm(images, desc=f"Processing {character} {'training' if is_train else 'validation'} images"):
            # Copy image
            shutil.copy2(img_path, target_dir / "images" / img_path.name)
            
            # Create YOLO format label
            # For now, we'll create a simple bounding box that covers the whole image
            # In a real scenario, you would need to annotate the actual bounding boxes
            label_path = target_dir / "labels" / f"{img_path.stem}.txt"
            with open(label_path, "w") as f:
                # Format: class_id x_center y_center width height
                # All values normalized to [0,1]
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
    
    def create_dataset_yaml(self):
        """Create dataset.yaml file for YOLO training."""
        yaml_data = {
            'path': str(self.output_dir.absolute()),
            'train': str(self.train_dir / "images"),
            'val': str(self.val_dir / "images"),
            'names': self.classes
        }
        
        with open(self.output_dir / "dataset.yaml", "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False)
    
    def run(self):
        """Run the dataset preparation process."""
        print("Starting dataset preparation...")
        self.split_dataset()
        self.create_dataset_yaml()
        print("\nDataset preparation complete!")
        print(f"Dataset saved in: {self.output_dir.absolute()}")
        print("\nDataset structure:")
        print(f"Training images: {len(list((self.train_dir / 'images').glob('*.jpg')))}")
        print(f"Validation images: {len(list((self.val_dir / 'images').glob('*.jpg')))}")
        print(f"Classes: {', '.join(self.classes)}")

if __name__ == "__main__":
    preparator = DatasetPreparator()
    preparator.run() 