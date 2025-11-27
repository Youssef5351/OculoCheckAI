"""
Cataract Detection Model - Separate Training
Uses your existing cataract dataset structure:
  processed_images/
    train/
    test/

This trains independently from your DR model.
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

class CataractPreprocessor:
    """Preprocessing for cataract images"""
    
    def __init__(self, image_size=224):
        self.image_size = image_size
    
    def preprocess(self, img_path):
        """Load and preprocess image"""
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        
        # Cataract images often need lens region focus
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2 * padding)
            h = min(img.shape[0] - y, h + 2 * padding)
            img = img[y:y+h, x:x+w]
        
        # Resize
        img = cv2.resize(img, (self.image_size, self.image_size))
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        return img

def scan_dataset(base_path):
    """
    Scan your folder structure and create labels
    Assumes structure:
      processed_images/
        train/
          cataract/  (or normal/, cataract_positive/, etc.)
          normal/
        test/
          cataract/
          normal/
    
    OR flat structure with filenames indicating class
    """
    base_path = Path(base_path)
    
    data = []
    
    print(f"\n🔍 Scanning dataset in: {base_path}")
    print("="*70)
    
    for split in ['train', 'test']:
        split_path = base_path / split
        
        if not split_path.exists():
            print(f"⚠️  Warning: {split_path} not found!")
            continue
        
        print(f"\n📂 Scanning {split}/")
        
        # Check if organized by folders (cataract/normal)
        subdirs = [d for d in split_path.iterdir() if d.is_dir()]
        
        if len(subdirs) > 0:
            # Folder-based organization
            print(f"   Found {len(subdirs)} class folders: {[d.name for d in subdirs]}")
            
            for class_dir in subdirs:
                class_name = class_dir.name.lower()
                
                # Determine label
                if 'cataract' in class_name or 'positive' in class_name or class_name == '1':
                    label = 1
                elif 'normal' in class_name or 'negative' in class_name or class_name == '0':
                    label = 0
                else:
                    print(f"   ⚠️  Unknown class: {class_name}, skipping...")
                    continue
                
                # Get all images
                image_files = list(class_dir.glob('*.jpg')) + \
                             list(class_dir.glob('*.jpeg')) + \
                             list(class_dir.glob('*.png'))
                
                for img_path in image_files:
                    data.append({
                        'image_path': str(img_path),
                        'label': label,
                        'split': split,
                        'class': 'cataract' if label == 1 else 'normal'
                    })
                
                print(f"   {class_dir.name}: {len(image_files)} images (label={label})")
        
        else:
            # Flat structure - try to infer from filenames
            print(f"   No class folders found, checking filenames...")
            
            image_files = list(split_path.glob('*.jpg')) + \
                         list(split_path.glob('*.jpeg')) + \
                         list(split_path.glob('*.png'))
            
            for img_path in image_files:
                filename = img_path.stem.lower()
                
                # Infer label from filename
                if 'cataract' in filename or 'positive' in filename or filename.startswith('1_'):
                    label = 1
                elif 'normal' in filename or 'negative' in filename or filename.startswith('0_'):
                    label = 0
                else:
                    # Default to checking if it contains numbers
                    print(f"   ⚠️  Cannot determine label for: {img_path.name}")
                    continue
                
                data.append({
                    'image_path': str(img_path),
                    'label': label,
                    'split': split,
                    'class': 'cataract' if label == 1 else 'normal'
                })
            
            print(f"   Found {len(image_files)} images")
    
    df = pd.DataFrame(data)
    
    if len(df) == 0:
        print("\n❌ ERROR: No images found!")
        print("\nPlease check your folder structure. Expected:")
        print("  processed_images/")
        print("    train/")
        print("      cataract/  (or 1/, positive/, etc.)")
        print("      normal/    (or 0/, negative/, etc.)")
        print("    test/")
        print("      cataract/")
        print("      normal/")
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
            class_name = 'Normal' if label == 0 else 'Cataract'
            print(f"  {class_name:10s}: {count:4d} ({pct:5.1f}%)")
    
    return df

class CataractDataGenerator(keras.utils.Sequence):
    """Data generator for cataract detection"""
    
    def __init__(self, df, preprocessor, batch_size=32, augment=False):
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.augment = augment
        
        print(f"\n🔍 Validating {len(df)} images...")
        valid_data = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Checking"):
            img_path = Path(row['image_path'])
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    valid_data.append(row)
        
        self.df = pd.DataFrame(valid_data).reset_index(drop=True)
        print(f"✅ {len(self.df)} valid images")
        
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        images = []
        labels = []
        
        for i in batch_indices:
            if i >= len(self.df):
                continue
            
            row = self.df.iloc[i]
            img = self.preprocessor.preprocess(Path(row['image_path']))
            
            if img is not None:
                # Augmentation
                if self.augment and np.random.random() > 0.5:
                    if np.random.random() > 0.5:
                        img = np.fliplr(img)
                    if np.random.random() > 0.5:
                        img = np.flipud(img)
                    if np.random.random() > 0.7:
                        img = np.rot90(img)
                    
                    # Brightness/contrast for cataract
                    if np.random.random() > 0.5:
                        alpha = np.random.uniform(0.8, 1.2)
                        beta = np.random.uniform(-0.1, 0.1)
                        img = np.clip(alpha * img + beta, 0, 1)
                
                images.append(img)
                labels.append(float(row['label']))
        
        if len(images) == 0:
            return np.zeros((1, 224, 224, 3)), np.array([0.0])
        
        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

def build_cataract_model():
    """Build EfficientNetB2 model for cataract detection"""
    
    base = EfficientNetB2(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    
    # Fine-tune from layer 100 onwards
    for layer in base.layers[:100]:
        layer.trainable = False
    for layer in base.layers[100:]:
        layer.trainable = True
    
    inputs = keras.Input(shape=(224, 224, 3))
    x = keras.applications.efficientnet.preprocess_input(inputs * 255.0)
    x = base(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return keras.Model(inputs, outputs, name='CataractDetector')

def train_model(train_gen, val_gen, epochs=40):
    """Train cataract detection model"""
    
    model = build_cataract_model()
    
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    print("\n" + "="*70)
    print("🚀 TRAINING CATARACT DETECTION MODEL")
    print("="*70)
    
    Path('models').mkdir(exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            'models/cataract_best.keras',
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_auc',
            patience=10,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def evaluate_model(model, test_df, preprocessor):
    """Evaluate on test set"""
    
    print("\n" + "="*70)
    print("📊 EVALUATING CATARACT MODEL")
    print("="*70)
    
    y_true = []
    y_pred = []
    
    print("\nGenerating predictions...")
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        img_path = Path(row['image_path'])
        img = preprocessor.preprocess(img_path)
        
        if img is not None:
            img_batch = np.expand_dims(img, axis=0)
            pred = model.predict(img_batch, verbose=0)[0][0]
            
            y_pred.append(pred)
            y_true.append(row['label'])
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred_binary = (y_pred >= optimal_threshold).astype(int)
    
    # Metrics
    print("\n" + classification_report(y_true, y_pred_binary,
                                       target_names=['Normal', 'Cataract'],
                                       digits=3))
    
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    
    acc = (tn + tp) / (tn + fp + fn + tp)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n📊 FINAL RESULTS:")
    print("="*70)
    print(f"Accuracy:    {acc:.3f} ({acc*100:.1f}%)")
    print(f"Sensitivity: {sens:.3f} ({sens*100:.1f}%)")
    print(f"Specificity: {spec:.3f} ({spec*100:.1f}%)")
    print(f"AUC:         {roc_auc:.3f}")
    print(f"Threshold:   {optimal_threshold:.4f}")
    print("="*70)
    
    # Save results
    results = {
        'accuracy': float(acc),
        'sensitivity': float(sens),
        'specificity': float(spec),
        'auc': float(roc_auc),
        'optimal_threshold': float(optimal_threshold)
    }
    
    with open('cataract_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n💾 Results saved: cataract_results.json")
    
    return results

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("👁️  CATARACT DETECTION - SEPARATE MODEL")
    print("="*70)
    
    # Step 1: Scan and organize dataset
    df = scan_dataset('processed_images')
    
    if df is None or len(df) == 0:
        print("\n❌ No data found! Please check your folder structure.")
        exit(1)
    
    # Save organized dataset
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    df.to_csv('data/processed/cataract_dataset.csv', index=False)
    print(f"\n💾 Dataset info saved: data/processed/cataract_dataset.csv")
    
    # Step 2: Split data
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    # Create validation from train (20%)
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(train_df, test_size=0.2, 
                                         random_state=42, 
                                         stratify=train_df['label'])
    
    print(f"\n📊 Final Split:")
    print(f"   Training:   {len(train_df)} images")
    print(f"   Validation: {len(val_df)} images")
    print(f"   Testing:    {len(test_df)} images")
    
    # Step 3: Create preprocessor and generators
    preprocessor = CataractPreprocessor(image_size=224)
    
    train_gen = CataractDataGenerator(train_df, preprocessor, batch_size=32, augment=True)
    val_gen = CataractDataGenerator(val_df, preprocessor, batch_size=32, augment=False)
    
    # Step 4: Train
    model, history = train_model(train_gen, val_gen, epochs=40)
    
    # Step 5: Evaluate
    results = evaluate_model(model, test_df, preprocessor)

    # Step 6: Save the model (THIS WAS MISSING)
    Path("models").mkdir(exist_ok=True)
    model.save("models/cataract_best.keras")
    print("\n💾 Model saved: models/cataract_best.keras")

    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📊 Results: cataract_results.json")
    print(f"\n🎯 This model is separate from your DR model.")
    print(f"   You now have:")
    print(f"   • models/eyemri_final_model.keras  (DR detection)")
    print(f"   • models/cataract_best.keras       (Cataract detection)")
    print("="*70)
