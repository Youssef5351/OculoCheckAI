import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import albumentations as A
from tqdm import tqdm
import matplotlib.pyplot as plt

class RetinalImagePreprocessor:
    """Preprocesses retinal fundus images"""
    
    def __init__(self, image_size=224):
        self.image_size = image_size
        
    def crop_black_borders(self, img):
        """Remove black borders from fundus images"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get bounding box of largest contour
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Add small padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2 * padding)
            h = min(img.shape[0] - y, h + 2 * padding)
            
            return img[y:y+h, x:x+w]
        
        return img
    
    def apply_clahe(self, img):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    def preprocess(self, img_path):
        """Full preprocessing pipeline"""
        # Read image
        img = cv2.imread(str(img_path))
        
        if img is None:
            return None
        
        # Crop black borders
        img = self.crop_black_borders(img)
        
        # Apply CLAHE for better contrast
        img = self.apply_clahe(img)
        
        # Resize
        img = cv2.resize(img, (self.image_size, self.image_size))
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        return img

def create_augmentation_pipeline(training=True):
    """Create data augmentation pipeline"""
    
    if training:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=45,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.CoarseDropout(
                max_holes=8,
                max_height=16,
                max_width=16,
                p=0.3
            )
        ])
    else:
        return None  # No augmentation for validation/test

def scan_image_directory(base_dir='retino'):
    """Scan the directory structure and create a dataframe"""
    
    base_dir = Path(base_dir)
    
    # Collect all image paths and labels
    data = []
    
    # Scan through train/valid/test folders
    for split in ['train', 'valid', 'test']:
        split_dir = base_dir / split
        
        if not split_dir.exists():
            print(f"⚠️  Directory not found: {split_dir}")
            continue
        
        # Scan DR and No_DR folders
        for label_name in ['DR', 'No_DR']:
            label_dir = split_dir / label_name
            
            if not label_dir.exists():
                continue
            
            # Get all images
            for img_path in label_dir.glob('*.png'):
                data.append({
                    'image_path': str(img_path),
                    'filename': img_path.name,
                    'id_code': img_path.stem,
                    'diagnosis': 1 if label_name == 'DR' else 0,  # Binary: 1=DR, 0=No_DR
                    'split': split
                })
            
            # Also check for jpg files
            for img_path in label_dir.glob('*.jpg'):
                data.append({
                    'image_path': str(img_path),
                    'filename': img_path.name,
                    'id_code': img_path.stem,
                    'diagnosis': 1 if label_name == 'DR' else 0,
                    'split': split
                })
    
    return pd.DataFrame(data)

def prepare_dataset(data_dir='retino', output_dir='data/processed'):
    """Prepare dataset from directory structure"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Scan directories
    print("📂 Scanning image directories...")
    df = scan_image_directory(data_dir)
    
    if len(df) == 0:
        print("❌ No images found! Check your directory structure.")
        return None, None, None
    
    print(f"\n✅ Found {len(df)} images")
    
    # Split by existing splits if available
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'valid'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    # If no pre-existing split, create one
    if len(val_df) == 0 and len(train_df) > 0:
        print("\n⚠️  No validation split found, creating one from training data...")
        train_df, val_df = train_test_split(
            train_df,
            test_size=0.2,
            random_state=42,
            stratify=train_df['diagnosis']
        )
    
    print(f"\n📊 Dataset Statistics:")
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    print(f"\n📈 Training set class distribution:")
    if len(train_df) > 0:
        print(train_df['diagnosis'].value_counts().sort_index())
        print(f"\nNo DR (0): {len(train_df[train_df['diagnosis']==0])}")
        print(f"DR (1): {len(train_df[train_df['diagnosis']==1])}")
    
    # Save splits
    if len(train_df) > 0:
        train_df.to_csv(output_dir / 'train.csv', index=False)
        print(f"✅ Saved train.csv")
    
    if len(val_df) > 0:
        val_df.to_csv(output_dir / 'val.csv', index=False)
        print(f"✅ Saved val.csv")
    
    if len(test_df) > 0:
        test_df.to_csv(output_dir / 'test.csv', index=False)
        print(f"✅ Saved test.csv")
    
    # Verify preprocessing on a sample image
    if len(train_df) > 0:
        preprocessor = RetinalImagePreprocessor(image_size=224)
        
        print("\n🔍 Verifying image preprocessing...")
        sample_img_path = Path(train_df.iloc[0]['image_path'])
        
        if sample_img_path.exists():
            processed = preprocessor.preprocess(sample_img_path)
            
            if processed is not None:
                print("✅ Image preprocessing working!")
                print(f"Processed image shape: {processed.shape}")
                print(f"Value range: [{processed.min():.3f}, {processed.max():.3f}]")
            else:
                print("❌ Image preprocessing failed!")
        else:
            print(f"⚠️  Sample image not found: {sample_img_path}")
    
    return train_df, val_df, test_df

def visualize_preprocessing(img_path, save_path='preprocessing_demo.png'):
    """Visualize preprocessing steps"""
    
    preprocessor = RetinalImagePreprocessor(image_size=224)
    
    # Read original
    original = cv2.imread(str(img_path))
    if original is None:
        print(f"❌ Could not read image: {img_path}")
        return
    
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Step by step
    cropped = preprocessor.crop_black_borders(cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    
    enhanced = preprocessor.apply_clahe(cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    
    resized = cv2.resize(enhanced, (224, 224))
    
    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(original)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(cropped)
    axes[1].set_title('Cropped')
    axes[1].axis('off')
    
    axes[2].imshow(enhanced)
    axes[2].set_title('CLAHE Enhanced')
    axes[2].axis('off')
    
    axes[3].imshow(resized)
    axes[3].set_title('Resized (224x224)')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    print("=" * 60)
    print("EyeMRI AI - Data Preprocessing")
    print("=" * 60)
    
    # Prepare dataset
    train_df, val_df, test_df = prepare_dataset()
    
    if train_df is not None and len(train_df) > 0:
        # Visualize preprocessing on first sample
        sample_img = Path(train_df.iloc[0]['image_path'])
        
        if sample_img.exists():
            visualize_preprocessing(sample_img)
        
        print("\n✅ Data preparation complete!")
        print("Next step: Run model training script")
    else:
        print("\n❌ No training data found. Please check your directory structure.")
        print("\nExpected structure:")
        print("retino/")
        print("  ├── train/")
        print("  │   ├── DR/")
        print("  │   └── No_DR/")
        print("  ├── valid/")
        print("  │   ├── DR/")
        print("  │   └── No_DR/")
        print("  └── test/")
        print("      ├── DR/")
        print("      └── No_DR/")