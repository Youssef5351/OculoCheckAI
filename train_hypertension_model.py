"""
Hypertension Detection Model - Separate Training
Uses the same structure as your cataract model
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from pathlib import Path
from tqdm import tqdm
import cv2
import json
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

class HypertensionPreprocessor:
    """Preprocessing for hypertension retinal images"""
    
    def __init__(self, image_size=224):
        self.image_size = image_size
    
    def preprocess(self, img_path):
        """Load and preprocess retinal image for hypertension detection"""
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Enhanced preprocessing for hypertension features
        # Focus on blood vessels and optic disc
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest reasonable contour
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > img.shape[1] * 0.3 and h > img.shape[0] * 0.3:
                    # Add padding
                    pad_x = max(10, int(w * 0.02))
                    pad_y = max(10, int(h * 0.02))
                    x = max(0, x - pad_x)
                    y = max(0, y - pad_y)
                    w = min(img.shape[1] - x, w + 2 * pad_x)
                    h = min(img.shape[0] - y, h + 2 * pad_y)
                    img = img[y:y+h, x:x+w]
                    break
        
        # CLAHE enhancement for better vessel visibility
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab_planes = list(cv2.split(lab))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Resize
        img = cv2.resize(img, (self.image_size, self.image_size))
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        return img

def scan_hypertension_dataset(base_path):
    """
    Scan hypertension dataset structure
    Assumes structure:
      hypertension_split/
        train/
          Normal/
          Hypertensive/
        val/
          Normal/ 
          Hypertensive/
        test/
          Normal/
          Hypertensive/
    """
    base_path = Path(base_path)
    
    data = []
    
    print(f"\n🔍 Scanning hypertension dataset in: {base_path}")
    print("="*70)
    
    for split in ['train', 'val', 'test']:
        split_path = base_path / split
        
        if not split_path.exists():
            print(f"⚠️  Warning: {split_path} not found!")
            continue
        
        print(f"\n📂 Scanning {split}/")
        
        # Check for class folders
        for class_name, label in [('Normal', 0), ('Hypertensive', 1)]:
            class_path = split_path / class_name
            
            if class_path.exists():
                image_files = list(class_path.glob('*.jpg')) + \
                             list(class_path.glob('*.jpeg')) + \
                             list(class_path.glob('*.png'))
                
                for img_path in image_files:
                    data.append({
                        'image_path': str(img_path),
                        'label': label,
                        'split': split,
                        'class': class_name.lower()
                    })
                
                print(f"   {class_name}: {len(image_files)} images")
            else:
                print(f"   ⚠️  {class_name} folder not found")
    
    df = pd.DataFrame(data)
    
    if len(df) == 0:
        print("\n❌ ERROR: No images found!")
        print("\nPlease check your folder structure. Expected:")
        print("  hypertension_split/")
        print("    train/")
        print("      Normal/")
        print("      Hypertensive/")
        print("    val/")
        print("      Normal/")
        print("      Hypertensive/")
        print("    test/")
        print("      Normal/")
        print("      Hypertensive/")
        return None
    
    print(f"\n✅ Total images found: {len(df)}")
    print("\n📊 Dataset Statistics:")
    print("="*70)
    
    for split in df['split'].unique():
        split_df = df[df['split'] == split]
        print(f"\n{split.upper()}:")
        print(f"  Total: {len(split_df)} images")
        for label in [0, 1]:
            count = len(split_df[split_df['label'] == label])
            pct = (count / len(split_df)) * 100
            class_name = 'Normal' if label == 0 else 'Hypertensive'
            print(f"  {class_name:12s}: {count:4d} ({pct:5.1f}%)")
    
    return df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("🫀 HYPERTENSION DETECTION - SEPARATE MODEL")
    print("="*70)
    print("🎯 Training style: Same as cataract model")
    print("="*70)
    
    # Step 1: Scan and organize dataset
    df = scan_hypertension_dataset('hypertension_split')
    
    if df is None or len(df) == 0:
        print("\n❌ No data found! Please check your folder structure.")
        exit(1)
    
    # Save organized dataset
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    df.to_csv('data/processed/hypertension_dataset.csv', index=False)
    print(f"\n💾 Dataset info saved: data/processed/hypertension_dataset.csv")
    
    # Step 2: Split data (using existing splits)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    # If no validation split, create from train
    if len(val_df) == 0 and len(train_df) > 0:
        train_df, val_df = train_test_split(train_df, test_size=0.2, 
                                           random_state=42, 
                                           stratify=train_df['label'])
    
    print(f"\n📊 Final Split:")
    print(f"   Training:   {len(train_df)} images")
    print(f"   Validation: {len(val_df)} images") 
    print(f"   Testing:    {len(test_df)} images")
    
    # Step 3: Create preprocessor and generators
    preprocessor = HypertensionPreprocessor(image_size=224)
    
    train_gen = HypertensionDataGenerator(train_df, preprocessor, batch_size=32, augment=True)
    val_gen = HypertensionDataGenerator(val_df, preprocessor, batch_size=32, augment=False)
    
    # Step 4: Train
    model, history = train_hypertension_model(train_gen, val_gen, epochs=50)
    
    # Step 5: Evaluate
    results = evaluate_hypertension_model(model, test_df, preprocessor)
    
    print("\n" + "="*70)
    print("✅ HYPERTENSION TRAINING COMPLETE!")
    print("="*70)
    print(f"\n💾 Model saved: models/hypertension_best.keras")
    print(f"📊 Results: hypertension_results.json")
    print(f"\n🎯 You now have separate models for:")
    print(f"   • Cataract detection:    models/cataract_best.keras")
    print(f"   • Hypertension detection: models/hypertension_best.keras")
    print(f"   • DR detection:          models/eyemri_final_model.keras")
    print("\n💡 Use each model independently for specific disease detection")
    print("="*70)