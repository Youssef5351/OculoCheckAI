"""
Test Your Trained DR Detection Models
Tests the models you've already trained and evaluates performance
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from tqdm import tqdm
import cv2
import json
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

class RetinalImagePreprocessor:
    """Same preprocessing as training"""
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

def test_single_model(model_path, test_df, preprocessor, model_name):
    """Test a single model"""
    
    print("\n" + "="*70)
    print(f"📊 TESTING: {model_name}")
    print("="*70)
    
    # Load model
    print(f"\n📂 Loading model: {model_path}")
    try:
        model = keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Get predictions
    y_true = []
    y_pred = []
    
    print(f"\n🔍 Testing on {len(test_df)} images...")
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
        img_path = Path(row['image_path'])
        
        if not img_path.exists():
            continue
            
        img = preprocessor.preprocess(img_path)
        
        if img is not None:
            img_batch = np.expand_dims(img, axis=0)
            pred = model.predict(img_batch, verbose=0)[0][0]
            
            y_pred.append(pred)
            y_true.append(float(row['diagnosis']))
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true) == 0:
        print("❌ No valid predictions!")
        return None
    
    # Calculate metrics
    print("\n📊 Calculating metrics...")
    
    # ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Binary predictions
    y_pred_binary = (y_pred >= optimal_threshold).astype(int)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = recall  # Same as recall
    
    # Print results
    print("\n" + "="*70)
    print(f"🎯 {model_name} - TEST RESULTS")
    print("="*70)
    
    print("\n📈 Performance Metrics:")
    print(f"  Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  AUC:         {roc_auc:.4f} ({roc_auc*100:.2f}%)")
    print(f"  Precision:   {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:      {recall:.4f} ({recall*100:.2f}%)")
    print(f"  Sensitivity: {sensitivity:.4f} ({sensitivity*100:.2f}%)")
    print(f"  Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"  F1-Score:    {f1:.4f}")
    
    print(f"\n🎚️  Optimal Threshold: {optimal_threshold:.4f}")
    
    print("\n📊 Confusion Matrix:")
    print(f"                Predicted")
    print(f"                No DR    DR")
    print(f"  Actual No DR   {tn:4d}  {fp:4d}")
    print(f"  Actual DR      {fn:4d}  {tp:4d}")
    
    print("\n" + classification_report(
        y_true, y_pred_binary, 
        target_names=['No DR', 'DR'], 
        digits=4
    ))
    
    # Save results
    results = {
        'model_name': model_name,
        'model_path': str(model_path),
        'test_samples': int(len(y_true)),
        'accuracy': float(accuracy),
        'auc': float(roc_auc),
        'precision': float(precision),
        'recall': float(recall),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'f1_score': float(f1),
        'optimal_threshold': float(optimal_threshold),
        'confusion_matrix': {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        }
    }
    
    return results, y_true, y_pred

def plot_roc_curve(results_list, save_path='test_results_roc.png'):
    """Plot ROC curves for all models"""
    
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for i, (results, y_true, y_pred) in enumerate(results_list):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(
            fpr, tpr, 
            color=colors[i % len(colors)],
            lw=2,
            label=f"{results['model_name']} (AUC = {roc_auc:.4f})"
        )
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.5000)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 ROC curve saved: {save_path}")
    plt.close()

def plot_confusion_matrices(results_list, save_path='test_results_confusion.png'):
    """Plot confusion matrices for all models"""
    
    n_models = len(results_list)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (results, y_true, y_pred) in enumerate(results_list):
        cm = confusion_matrix(
            y_true, 
            (y_pred >= results['optimal_threshold']).astype(int)
        )
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['No DR', 'DR'],
            yticklabels=['No DR', 'DR'],
            ax=axes[idx],
            cbar=False
        )
        
        axes[idx].set_title(
            f"{results['model_name']}\nAcc: {results['accuracy']:.2%}", 
            fontsize=12, 
            fontweight='bold'
        )
        axes[idx].set_ylabel('True Label', fontsize=10)
        axes[idx].set_xlabel('Predicted Label', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Confusion matrices saved: {save_path}")
    plt.close()

def test_ensemble(model_paths, test_df, preprocessor, model_names):
    """Test ensemble of multiple models"""
    
    print("\n" + "="*70)
    print("🎯 ENSEMBLE TESTING - Combining Multiple Models")
    print("="*70)
    
    all_predictions = []
    y_true = None
    
    for model_path, model_name in zip(model_paths, model_names):
        print(f"\n📂 Loading {model_name}...")
        
        try:
            model = keras.models.load_model(model_path)
            
            y_t = []
            y_p = []
            
            for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"  {model_name}"):
                img_path = Path(row['image_path'])
                
                if not img_path.exists():
                    continue
                    
                img = preprocessor.preprocess(img_path)
                
                if img is not None:
                    img_batch = np.expand_dims(img, axis=0)
                    pred = model.predict(img_batch, verbose=0)[0][0]
                    
                    y_p.append(pred)
                    y_t.append(float(row['diagnosis']))
            
            all_predictions.append(np.array(y_p))
            
            if y_true is None:
                y_true = np.array(y_t)
        
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")
            continue
    
    if len(all_predictions) == 0:
        print("❌ No models could be loaded!")
        return None
    
    # Average predictions
    ensemble_pred = np.mean(all_predictions, axis=0)
    
    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(y_true, ensemble_pred)
    roc_auc = auc(fpr, tpr)
    
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred_binary = (ensemble_pred >= optimal_threshold).astype(int)
    
    # Metrics
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print("\n" + "="*70)
    print("🎯 ENSEMBLE RESULTS")
    print("="*70)
    
    print(f"\n📊 Combined {len(all_predictions)} models:")
    for name in model_names:
        print(f"  • {name}")
    
    print("\n📈 Performance Metrics:")
    print(f"  Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  AUC:         {roc_auc:.4f} ({roc_auc*100:.2f}%)")
    print(f"  Precision:   {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:      {recall:.4f} ({recall*100:.2f}%)")
    print(f"  Sensitivity: {recall:.4f} ({recall*100:.2f}%)")
    print(f"  Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"  F1-Score:    {f1:.4f}")
    
    results = {
        'model_name': 'Ensemble',
        'num_models': len(all_predictions),
        'model_names': model_names,
        'test_samples': int(len(y_true)),
        'accuracy': float(accuracy),
        'auc': float(roc_auc),
        'precision': float(precision),
        'recall': float(recall),
        'sensitivity': float(recall),
        'specificity': float(specificity),
        'f1_score': float(f1),
        'optimal_threshold': float(optimal_threshold)
    }
    
    return results, y_true, ensemble_pred

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("🧪 TESTING DIABETIC RETINOPATHY DETECTION MODELS")
    print("="*70)
    
    # Load test data
    print("\n📂 Loading test dataset...")
    test_csv = 'data/processed/test.csv'
    
    if not Path(test_csv).exists():
        # Try validation set if test doesn't exist
        test_csv = 'data/processed/val.csv'
        print(f"⚠️  Test set not found, using validation set: {test_csv}")
    
    test_df = pd.read_csv(test_csv)
    print(f"✅ Loaded {len(test_df)} test images")
    
    # Initialize preprocessor
    preprocessor = RetinalImagePreprocessor(image_size=224)
    
    # Define models to test
    models_to_test = [
        {
            'path': 'models/eyemri_final_model.keras',
            'name': 'MobileNetV2 (Original)'
        },
        {
            'path': 'models/ensemble_EfficientNetB0.keras',
            'name': 'EfficientNetB0'
        },
        {
            'path': 'models/ensemble_ResNet50.keras',
            'name': 'ResNet50'
        }
    ]
    
    # Test individual models
    all_results = []
    
    for model_info in models_to_test:
        model_path = Path(model_info['path'])
        
        if not model_path.exists():
            print(f"\n⚠️  Model not found: {model_path}")
            continue
        
        result = test_single_model(
            model_path,
            test_df,
            preprocessor,
            model_info['name']
        )
        
        if result is not None:
            all_results.append(result)
    
    # Test ensemble if multiple models available
    available_models = [
        (Path(m['path']), m['name']) 
        for m in models_to_test 
        if Path(m['path']).exists()
    ]
    
    if len(available_models) >= 2:
        print("\n" + "="*70)
        print("🎯 Testing Ensemble (combining all models)...")
        print("="*70)
        
        model_paths = [m[0] for m in available_models]
        model_names = [m[1] for m in available_models]
        
        ensemble_result = test_ensemble(
            model_paths,
            test_df,
            preprocessor,
            model_names
        )
        
        if ensemble_result is not None:
            all_results.append(ensemble_result)
    
    # Save all results
    if len(all_results) > 0:
        output_file = 'test_results_summary.json'
        
        results_dict = {}
        for i, (results, _, _) in enumerate(all_results):
            results_dict[results['model_name']] = results
        
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        print("\n" + "="*70)
        print("📊 SUMMARY OF ALL MODELS")
        print("="*70)
        
        print(f"\n{'Model':<30} {'Accuracy':<12} {'AUC':<12} {'Sensitivity':<12} {'Specificity':<12}")
        print("-" * 78)
        
        for results, _, _ in all_results:
            print(f"{results['model_name']:<30} "
                  f"{results['accuracy']:.2%}     "
                  f"{results['auc']:.4f}     "
                  f"{results['sensitivity']:.2%}     "
                  f"{results['specificity']:.2%}")
        
        print("\n" + "="*70)
        print(f"💾 Results saved: {output_file}")
        
        # Generate visualizations
        print("\n📊 Generating visualizations...")
        plot_roc_curve(all_results)
        plot_confusion_matrices(all_results)
        
        print("\n✅ Testing complete!")
        print("="*70)
    
    else:
        print("\n❌ No models were successfully tested!")
        print("\nMake sure you have trained models in the 'models/' directory:")
        print("  • models/eyemri_final_model.keras")
        print("  • models/ensemble_EfficientNetB0.keras")
        print("  • models/ensemble_ResNet50.keras")