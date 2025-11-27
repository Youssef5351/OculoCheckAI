"""
Train Multiple Models and Ensemble Them
This typically gives 3-5% accuracy boost
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from pathlib import Path
from tqdm import tqdm
import cv2
import json

class RetinalImagePreprocessor:
    def __init__(self, image_size=224):
        self.image_size = image_size
        
    def preprocess(self, img_path):
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            padding = 5
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2 * padding)
            h = min(img.shape[0] - y, h + 2 * padding)
            img = img[y:y+h, x:x+w]
        
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0
        return img

class DataGenerator(keras.utils.Sequence):
    def __init__(self, df, preprocessor, batch_size=32, augment=False):
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.augment = augment
        
        print(f"🔍 Loading {len(df)} images...")
        valid_data = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Checking"):
            img_path = Path(row['image_path'])
            if img_path.exists() and cv2.imread(str(img_path)) is not None:
                valid_data.append(row)
        
        self.df = pd.DataFrame(valid_data).reset_index(drop=True)
        print(f"✅ {len(self.df)} valid images")
        
        self.indices = np.arange(len(self.df))
        np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        images, labels = [], []
        for i in batch_indices:
            if i >= len(self.df):
                continue
            row = self.df.iloc[i]
            img = self.preprocessor.preprocess(Path(row['image_path']))
            
            if img is not None:
                if self.augment and np.random.random() > 0.5:
                    if np.random.random() > 0.5:
                        img = np.fliplr(img)
                    if np.random.random() > 0.5:
                        img = np.flipud(img)
                    if np.random.random() > 0.5:
                        img = np.rot90(img)
                
                images.append(img)
                labels.append(float(row['diagnosis']))
        
        if len(images) == 0:
            return np.zeros((1, 224, 224, 3)), np.array([0.0])
        
        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

def build_mobilenet():
    """Model 1: MobileNetV2"""
    base = MobileNetV2(include_top=False, weights='imagenet', 
                       input_shape=(224, 224, 3), pooling='avg')
    base.trainable = True
    
    inputs = keras.Input(shape=(224, 224, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return keras.Model(inputs, outputs, name='MobileNetV2')

def build_efficientnet():
    """Model 2: EfficientNetB0"""
    base = EfficientNetB0(include_top=False, weights='imagenet',
                          input_shape=(224, 224, 3), pooling='avg')
    base.trainable = True
    
    inputs = keras.Input(shape=(224, 224, 3))
    x = keras.applications.efficientnet.preprocess_input(inputs * 255.0)
    x = base(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return keras.Model(inputs, outputs, name='EfficientNetB0')

def build_resnet():
    """Model 3: ResNet50"""
    base = ResNet50(include_top=False, weights='imagenet',
                    input_shape=(224, 224, 3), pooling='avg')
    base.trainable = True
    
    inputs = keras.Input(shape=(224, 224, 3))
    x = keras.applications.resnet50.preprocess_input(inputs * 255.0)
    x = base(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return keras.Model(inputs, outputs, name='ResNet50')

def train_single_model(model, model_name, train_gen, val_gen, epochs=25):
    """Train one model"""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    model.compile(
        optimizer=keras.optimizers.Adam(5e-5),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[
            ModelCheckpoint(
                f'models/ensemble_{model_name}.keras',
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_auc',
                patience=8,
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                verbose=1
            )
        ],
        verbose=2
    )
    
    return model

def evaluate_ensemble(models, val_df, preprocessor):
    """Evaluate ensemble of models"""
    print("\n" + "="*60)
    print("ENSEMBLE EVALUATION")
    print("="*60)
    
    y_true = []
    predictions_per_model = [[] for _ in models]
    
    print("\nGetting predictions from all models...")
    for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
        img_path = Path(row['image_path'])
        img = preprocessor.preprocess(img_path)
        
        if img is not None:
            img_batch = np.expand_dims(img, axis=0)
            
            for i, model in enumerate(models):
                pred = model.predict(img_batch, verbose=0)[0][0]
                predictions_per_model[i].append(pred)
            
            y_true.append(row['diagnosis'])
    
    y_true = np.array(y_true)
    
    # Ensemble: Average predictions
    ensemble_preds = np.mean([np.array(p) for p in predictions_per_model], axis=0)
    
    # Find optimal threshold
    from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
    
    fpr, tpr, thresholds = roc_curve(y_true, ensemble_preds)
    roc_auc = auc(fpr, tpr)
    
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred = (ensemble_preds >= optimal_threshold).astype(int)
    
    # Metrics
    print("\n" + classification_report(y_true, y_pred, 
                                       target_names=['No DR', 'DR'], 
                                       digits=3))
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = (tn + tp) / (tn + fp + fn + tp)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    
    print(f"\n📊 ENSEMBLE RESULTS:")
    print("="*60)
    print(f"Accuracy:    {acc:.3f} ({acc*100:.1f}%)")
    print(f"Sensitivity: {sens:.3f} ({sens*100:.1f}%)")
    print(f"Specificity: {spec:.3f} ({spec*100:.1f}%)")
    print(f"AUC:         {roc_auc:.3f}")
    print(f"Threshold:   {optimal_threshold:.4f}")
    print("="*60)
    
    # Compare with individual models
    print("\n📊 Individual Model Performance:")
    print("-"*60)
    for i, preds in enumerate(predictions_per_model):
        preds = np.array(preds)
        fpr_i, tpr_i, thresh_i = roc_curve(y_true, preds)
        auc_i = auc(fpr_i, tpr_i)
        j_i = tpr_i - fpr_i
        opt_thresh_i = thresh_i[np.argmax(j_i)]
        y_pred_i = (preds >= opt_thresh_i).astype(int)
        acc_i = (y_true == y_pred_i).mean()
        print(f"Model {i+1}: Accuracy={acc_i:.3f}, AUC={auc_i:.3f}")
    
    print(f"\n✅ Ensemble: Accuracy={acc:.3f}, AUC={roc_auc:.3f}")
    print(f"   Improvement: +{(acc - max([((y_true == (np.array(p) >= thresholds[np.argmax(roc_curve(y_true, np.array(p))[1] - roc_curve(y_true, np.array(p))[0])]).astype(int)).mean()) for p in predictions_per_model]))*100:.1f}%")
    
    # Save ensemble config
    config = {
        'optimal_threshold': float(optimal_threshold),
        'auc': float(roc_auc),
        'accuracy': float(acc),
        'sensitivity': float(sens),
        'specificity': float(spec),
        'num_models': len(models)
    }
    
    with open('ensemble_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\n💾 Ensemble config saved: ensemble_config.json")
    
    return roc_auc, acc

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎯 ENSEMBLE TRAINING - Boost Your Accuracy!")
    print("="*60)
    
    # Load data
    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')
    
    preprocessor = RetinalImagePreprocessor()
    
    train_gen = DataGenerator(train_df, preprocessor, batch_size=32, augment=True)
    val_gen = DataGenerator(val_df, preprocessor, batch_size=32, augment=False)
    
    # Train 3 different models
    print("\n🚀 Training 3 diverse models...")
    print("This will take ~3 hours total")
    
    models = []
    
    # Model 1: MobileNetV2 (already trained - can skip)
    print("\nℹ️  Using your existing MobileNetV2 model")
    model1 = keras.models.load_model('models/eyemri_final_model.keras')
    models.append(model1)
    
    # Model 2: EfficientNetB0
    model2 = build_efficientnet()
    model2 = train_single_model(model2, 'EfficientNetB0', train_gen, val_gen)
    models.append(model2)
    
    # Model 3: ResNet50
    model3 = build_resnet()
    model3 = train_single_model(model3, 'ResNet50', train_gen, val_gen)
    models.append(model3)
    
    # Evaluate ensemble
    ensemble_auc, ensemble_acc = evaluate_ensemble(models, val_df, preprocessor)
    
    print("\n" + "="*60)
    print("✅ ENSEMBLE TRAINING COMPLETE!")
    print("="*60)
    print(f"\n🎯 Expected improvement: 3-5% accuracy boost")
    print(f"📊 Final Ensemble Performance: {ensemble_acc*100:.1f}% accuracy, {ensemble_auc:.3f} AUC")
    print(f"\n💡 To use ensemble in production:")
    print(f"   1. Load all 3 models")
    print(f"   2. Get predictions from each")
    print(f"   3. Average the predictions")
    print(f"   4. Use threshold from ensemble_config.json")