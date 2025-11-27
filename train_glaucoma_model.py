"""
Glaucoma Detection: Train Multiple Models & Compare
Similar to your diabetic retinopathy ensemble approach

This script:
1. Trains 5 different architectures
2. Compares their individual performance
3. Creates an ensemble from the best ones
4. Shows you which single model is best
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import (
    EfficientNetB0, EfficientNetB2, EfficientNetB3,
    MobileNetV2, DenseNet121, ResNet50
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from pathlib import Path
from tqdm import tqdm
import cv2
import json
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)

class GlaucomaPreprocessor:
    """Same preprocessing as your working model"""
    
    def __init__(self, image_size=224):
        self.image_size = image_size
    
    def preprocess(self, img_path):
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        
        # Crop to retinal region
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
        
        # CLAHE enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Resize and normalize
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0
        
        return img

def scan_glaucoma_dataset(base_path):
    """Scan dataset"""
    base_path = Path(base_path)
    
    print("\n" + "="*70)
    print("📂 SCANNING GLAUCOMA DATASET")
    print("="*70)
    
    all_data = {'train': [], 'validate': [], 'test': []}
    
    for split in ['train', 'validate', 'test']:
        split_path = base_path / split
        
        if not split_path.exists():
            continue
        
        print(f"\n📁 {split}/")
        
        for class_name in ['NRG', 'RG']:
            class_path = split_path / class_name
            
            if not class_path.exists():
                continue
            
            label = 0 if class_name == 'NRG' else 1
            
            image_files = (
                list(class_path.glob('*.jpg')) + 
                list(class_path.glob('*.jpeg')) + 
                list(class_path.glob('*.png')) +
                list(class_path.glob('*.bmp'))
            )
            
            for img_path in image_files:
                all_data[split].append({
                    'image_path': str(img_path),
                    'label': label,
                    'class': 'Glaucoma' if label == 1 else 'Normal'
                })
            
            print(f"   {class_name}: {len(image_files)} images")
    
    train_df = pd.DataFrame(all_data['train'])
    val_df = pd.DataFrame(all_data['validate'])
    test_df = pd.DataFrame(all_data['test'])
    
    return train_df, val_df, test_df

class DataGenerator(keras.utils.Sequence):
    """Data generator with validation"""
    
    def __init__(self, df, preprocessor, batch_size=32, augment=False):
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.augment = augment
        
        print(f"🔍 Validating {len(df)} images...")
        valid_data = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Checking"):
            img_path = Path(row['image_path'])
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    valid_data.append(row)
        
        self.df = pd.DataFrame(valid_data).reset_index(drop=True)
        print(f"✅ {len(self.df)} valid images\n")
        
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def augment_image(self, img):
        """Simple augmentation"""
        if np.random.random() > 0.5:
            img = np.fliplr(img)
        if np.random.random() > 0.5:
            img = np.flipud(img)
        if np.random.random() > 0.7:
            angle = np.random.choice([90, 180, 270])
            img = np.rot90(img, k=angle//90)
        if np.random.random() > 0.5:
            alpha = np.random.uniform(0.85, 1.15)
            beta = np.random.uniform(-0.08, 0.08)
            img = np.clip(alpha * img + beta, 0, 1)
        return img
    
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
                if self.augment:
                    img = self.augment_image(img)
                
                images.append(img)
                labels.append(float(row['label']))
        
        if len(images) == 0:
            return np.zeros((1, 224, 224, 3)), np.array([0.0])
        
        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

def build_efficientnetb0():
    """Model 1: EfficientNetB0 (Lightweight & Fast)"""
    base = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    
    for layer in base.layers[:80]:
        layer.trainable = False
    for layer in base.layers[80:]:
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
    
    return keras.Model(inputs, outputs, name='EfficientNetB0')

def build_efficientnetb2():
    """Model 2: EfficientNetB2 (Your current best model)"""
    base = EfficientNetB2(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    
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
    
    return keras.Model(inputs, outputs, name='EfficientNetB2')

def build_efficientnetb3():
    """Model 3: EfficientNetB3 (Larger capacity)"""
    base = EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    
    for layer in base.layers[:120]:
        layer.trainable = False
    for layer in base.layers[120:]:
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
    
    return keras.Model(inputs, outputs, name='EfficientNetB3')

def build_mobilenetv2():
    """Model 4: MobileNetV2 (Very fast inference)"""
    base = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    
    for layer in base.layers[:80]:
        layer.trainable = False
    for layer in base.layers[80:]:
        layer.trainable = True
    
    inputs = keras.Input(shape=(224, 224, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return keras.Model(inputs, outputs, name='MobileNetV2')

def build_densenet121():
    """Model 5: DenseNet121 (Dense connections)"""
    base = DenseNet121(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    
    for layer in base.layers[:200]:
        layer.trainable = False
    for layer in base.layers[200:]:
        layer.trainable = True
    
    inputs = keras.Input(shape=(224, 224, 3))
    x = keras.applications.densenet.preprocess_input(inputs * 255.0)
    x = base(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return keras.Model(inputs, outputs, name='DenseNet121')

# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def train_single_model(model, model_name, train_gen, val_gen, epochs=40):
    """Train one model"""
    
    print("\n" + "="*70)
    print(f"🚀 TRAINING: {model_name}")
    print("="*70)
    
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    
    Path('models/ensemble').mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            f'models/ensemble/glaucoma_{model_name}.keras',
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

def evaluate_single_model(model, model_name, test_df, preprocessor):
    """Evaluate one model"""
    
    print(f"\n📊 Evaluating {model_name}...")
    
    y_true = []
    y_pred = []
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Testing {model_name}"):
        img_path = Path(row['image_path'])
        img = preprocessor.preprocess(img_path)
        
        if img is not None:
            img_batch = np.expand_dims(img, axis=0)
            pred = model.predict(img_batch, verbose=0)[0][0]
            
            y_pred.append(pred)
            y_true.append(row['label'])
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred_binary = (y_pred >= optimal_threshold).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'model_name': model_name,
        'predictions': y_pred,
        'accuracy': accuracy,
        'auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_score': f1,
        'optimal_threshold': optimal_threshold,
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    }

def evaluate_ensemble(all_results, y_true):
    """Evaluate ensemble of all models"""
    
    print("\n" + "="*70)
    print("🎯 ENSEMBLE EVALUATION")
    print("="*70)
    
    # Average predictions from all models
    all_predictions = [result['predictions'] for result in all_results]
    ensemble_pred = np.mean(all_predictions, axis=0)
    
    # Calculate ensemble metrics
    fpr, tpr, thresholds = roc_curve(y_true, ensemble_pred)
    roc_auc = auc(fpr, tpr)
    
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred_binary = (ensemble_pred >= optimal_threshold).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_score': f1,
        'optimal_threshold': optimal_threshold,
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    }

def print_comparison(all_results, ensemble_results):
    """Print comparison table"""
    
    print("\n" + "="*70)
    print("📊 MODEL COMPARISON")
    print("="*70)
    print(f"\n{'Model':<20} {'Accuracy':<12} {'AUC':<12} {'Sensitivity':<12} {'Specificity':<12}")
    print("-"*70)
    
    # Individual models
    for result in all_results:
        print(f"{result['model_name']:<20} "
              f"{result['accuracy']:<12.4f} "
              f"{result['auc']:<12.4f} "
              f"{result['sensitivity']:<12.4f} "
              f"{result['specificity']:<12.4f}")
    
    # Ensemble
    print("-"*70)
    print(f"{'ENSEMBLE':<20} "
          f"{ensemble_results['accuracy']:<12.4f} "
          f"{ensemble_results['auc']:<12.4f} "
          f"{ensemble_results['sensitivity']:<12.4f} "
          f"{ensemble_results['specificity']:<12.4f}")
    print("="*70)
    
    # Find best individual model
    best_model = max(all_results, key=lambda x: x['accuracy'])
    
    print(f"\n🏆 BEST INDIVIDUAL MODEL: {best_model['model_name']}")
    print(f"   Accuracy: {best_model['accuracy']:.4f} ({best_model['accuracy']*100:.2f}%)")
    print(f"   AUC:      {best_model['auc']:.4f} ({best_model['auc']*100:.2f}%)")
    
    print(f"\n🎯 ENSEMBLE PERFORMANCE:")
    print(f"   Accuracy: {ensemble_results['accuracy']:.4f} ({ensemble_results['accuracy']*100:.2f}%)")
    print(f"   AUC:      {ensemble_results['auc']:.4f} ({ensemble_results['auc']*100:.2f}%)")
    
    improvement = (ensemble_results['accuracy'] - best_model['accuracy']) * 100
    print(f"\n📈 Improvement: {improvement:+.2f}% accuracy over best single model")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("👁️  GLAUCOMA ENSEMBLE TRAINING & COMPARISON")
    print("="*70)
    print("\nThis script will:")
    print("  1. Train 5 different model architectures")
    print("  2. Compare their individual performance")
    print("  3. Create an ensemble from all models")
    print("  4. Show you which model is best")
    print("="*70)
    
    # Configuration
    DATASET_PATH = 'glaucoma_dataset'
    BATCH_SIZE = 32
    EPOCHS = 40
    
    # Models to train
    MODEL_BUILDERS = [
        ('EfficientNetB0', build_efficientnetb0),
        ('EfficientNetB2', build_efficientnetb2),
        ('EfficientNetB3', build_efficientnetb3),
        ('MobileNetV2', build_mobilenetv2),
        ('DenseNet121', build_densenet121),
    ]
    
    print(f"\n⚙️  Configuration:")
    print(f"   Dataset:    {DATASET_PATH}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Epochs:     {EPOCHS}")
    print(f"   Models:     {len(MODEL_BUILDERS)}")
    
    # Load dataset
    train_df, val_df, test_df = scan_glaucoma_dataset(DATASET_PATH)
    
    if len(train_df) == 0:
        print("\n❌ No training data found!")
        exit(1)
    
    # Create preprocessor and generators
    preprocessor = GlaucomaPreprocessor()
    
    train_gen = DataGenerator(train_df, preprocessor, batch_size=BATCH_SIZE, augment=True)
    val_gen = DataGenerator(val_df, preprocessor, batch_size=BATCH_SIZE, augment=False)
    
    # Train all models
    print("\n" + "="*70)
    print("🚀 TRAINING ALL MODELS")
    print("="*70)
    print(f"⏱️  Estimated time: ~{len(MODEL_BUILDERS) * 2} hours")
    
    trained_models = []
    
    for model_name, build_func in MODEL_BUILDERS:
        print(f"\n{'='*70}")
        print(f"Training {model_name} ({MODEL_BUILDERS.index((model_name, build_func)) + 1}/{len(MODEL_BUILDERS)})")
        print(f"{'='*70}")
        
        model = build_func()
        model, history = train_single_model(model, model_name, train_gen, val_gen, epochs=EPOCHS)
        trained_models.append((model_name, model))
    
    # Evaluate all models
    print("\n" + "="*70)
    print("📊 EVALUATING ALL MODELS")
    print("="*70)
    
    all_results = []
    y_true = None
    
    for model_name, model in trained_models:
        result = evaluate_single_model(model, model_name, test_df, preprocessor)
        all_results.append(result)
        
        if y_true is None:
            y_true = []
            for _, row in test_df.iterrows():
                img_path = Path(row['image_path'])
                img = preprocessor.preprocess(img_path)
                if img is not None:
                    y_true.append(row['label'])
            y_true = np.array(y_true)
    
    # Evaluate ensemble
    ensemble_results = evaluate_ensemble(all_results, y_true)
    
    # Print comparison
    print_comparison(all_results, ensemble_results)
    
    # Save results
    final_results = {
        'individual_models': [
            {
                'name': r['model_name'],
                'accuracy': float(r['accuracy']),
                'auc': float(r['auc']),
                'sensitivity': float(r['sensitivity']),
                'specificity': float(r['specificity']),
                'f1_score': float(r['f1_score']),
                'threshold': float(r['optimal_threshold'])
            }
            for r in all_results
        ],
        'ensemble': {
            'accuracy': float(ensemble_results['accuracy']),
            'auc': float(ensemble_results['auc']),
            'sensitivity': float(ensemble_results['sensitivity']),
            'specificity': float(ensemble_results['specificity']),
            'f1_score': float(ensemble_results['f1_score']),
            'threshold': float(ensemble_results['optimal_threshold']),
            'num_models': len(MODEL_BUILDERS)
        }
    }
    
    with open('glaucoma_ensemble_results.json', 'w') as f:
        json.dump(final_results, f, indent=4)
    
    print("\n" + "="*70)
    print("✅ ENSEMBLE TRAINING COMPLETE!")
    print("="*70)
    print(f"\n💾 Models saved in: models/ensemble/")
    print(f"📊 Results saved: glaucoma_ensemble_results.json")
    print("\n💡 Recommendation:")
    
    best_model = max(all_results, key=lambda x: x['accuracy'])
    if ensemble_results['accuracy'] > best_model['accuracy']:
        print(f"   Use ENSEMBLE for best accuracy: {ensemble_results['accuracy']*100:.2f}%")
    else:
        print(f"   Use {best_model['model_name']} for best accuracy: {best_model['accuracy']*100:.2f}%")
    
    print("="*70)