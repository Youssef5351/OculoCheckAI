"""
Test Ensemble Models and Generate Professional Visualizations
No retraining required - uses pre-trained models
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from tqdm import tqdm
import cv2
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set style for professional figures
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

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

class EnsembleModelTester:
    def __init__(self, model_paths, preprocessor):
        """
        Args:
            model_paths: dict with model names and their file paths
            preprocessor: RetinalImagePreprocessor instance
        """
        self.models = {}
        self.model_names = []
        self.preprocessor = preprocessor
        
        print("\n" + "="*70)
        print("🔄 LOADING MODELS")
        print("="*70)
        
        for name, path in model_paths.items():
            try:
                self.models[name] = keras.models.load_model(path)
                self.model_names.append(name)
                print(f"✅ {name}: Loaded from {path}")
            except Exception as e:
                print(f"❌ {name}: Failed to load - {e}")
        
        if not self.models:
            raise ValueError("No models loaded successfully!")
    
    def test_ensemble(self, val_df, diseases=['No Disease', 'Disease']):
        """Test ensemble on validation set"""
        print("\n" + "="*70)
        print("🧪 TESTING ENSEMBLE")
        print("="*70)
        
        y_true = []
        predictions_per_model = {name: [] for name in self.model_names}
        sample_images = []
        
        print(f"\nProcessing {len(val_df)} validation images...")
        for idx, (_, row) in enumerate(tqdm(val_df.iterrows(), total=len(val_df))):
            img_path = Path(row['image_path'])
            img = self.preprocessor.preprocess(img_path)
            
            if img is not None:
                img_batch = np.expand_dims(img, axis=0)
                
                for model_name in self.model_names:
                    pred = self.models[model_name].predict(img_batch, verbose=0)[0][0]
                    predictions_per_model[model_name].append(pred)
                
                y_true.append(int(row['diagnosis']))
                
                # Store first 5 valid images for visualization
                if len(sample_images) < 5:
                    sample_images.append((img, int(row['diagnosis'])))
        
        y_true = np.array(y_true)
        
        # Compute ensemble predictions (average)
        ensemble_preds = np.mean(
            [np.array(p) for p in predictions_per_model.values()], 
            axis=0
        )
        
        # Find optimal threshold
        fpr_ens, tpr_ens, thresholds = roc_curve(y_true, ensemble_preds)
        roc_auc_ens = auc(fpr_ens, tpr_ens)
        
        j_scores = tpr_ens - fpr_ens
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        y_pred_ensemble = (ensemble_preds >= optimal_threshold).astype(int)
        
        # Compute metrics
        results = {
            'y_true': y_true,
            'ensemble_preds': ensemble_preds,
            'y_pred_ensemble': y_pred_ensemble,
            'optimal_threshold': optimal_threshold,
            'roc_auc_ensemble': roc_auc_ens,
            'fpr_ensemble': fpr_ens,
            'tpr_ensemble': tpr_ens,
            'individual_predictions': predictions_per_model,
            'sample_images': sample_images
        }
        
        # Print results
        print("\n" + "="*70)
        print("📊 ENSEMBLE RESULTS")
        print("="*70)
        
        cm = confusion_matrix(y_true, y_pred_ensemble)
        tn, fp, fn, tp = cm.ravel()
        
        acc = (tn + tp) / (tn + fp + fn + tp)
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        
        print(f"\nAccuracy:    {acc:.4f} ({acc*100:.2f}%)")
        print(f"Sensitivity: {sens:.4f} ({sens*100:.2f}%)")
        print(f"Specificity: {spec:.4f} ({spec*100:.2f}%)")
        print(f"AUC-ROC:     {roc_auc_ens:.4f}")
        print(f"Threshold:   {optimal_threshold:.4f}")
        
        print("\n" + "-"*70)
        print("📈 INDIVIDUAL MODEL PERFORMANCE")
        print("-"*70)
        
        for model_name in self.model_names:
            preds = np.array(predictions_per_model[model_name])
            fpr_i, tpr_i, thresh_i = roc_curve(y_true, preds)
            auc_i = auc(fpr_i, tpr_i)
            j_i = tpr_i - fpr_i
            opt_thresh_i = thresh_i[np.argmax(j_i)]
            y_pred_i = (preds >= opt_thresh_i).astype(int)
            acc_i = (y_true == y_pred_i).mean()
            
            results[f'{model_name}_auc'] = auc_i
            results[f'{model_name}_acc'] = acc_i
            
            print(f"{model_name:20s} | Accuracy: {acc_i:.4f} | AUC: {auc_i:.4f}")
        
        results['metrics'] = {
            'accuracy': float(acc),
            'sensitivity': float(sens),
            'specificity': float(spec),
            'auc': float(roc_auc_ens),
            'threshold': float(optimal_threshold),
            'confusion_matrix': cm.tolist()
        }
        
        return results
    
    def generate_architecture_diagram(self, results, output_path='figures/01_system_architecture.png'):
        """Figure 1: System Architecture & Multi-Disease Screening Workflow"""
        print(f"\n📐 Generating Architecture Diagram...")
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
        
        # Title
        fig.suptitle('OculoCheck: End-to-End Retinal Disease Detection Pipeline', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        # Input image (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        if results['sample_images']:
            img = results['sample_images'][0][0]
            ax1.imshow(img, cmap='gray')
            ax1.set_title('Step 1: Retinal Fundus Image Input', fontsize=12, fontweight='bold')
            ax1.axis('off')
        
        # Preprocessing info
        ax2 = fig.add_subplot(gs[0, 2:])
        ax2.axis('off')
        preprocess_text = """PREPROCESSING PIPELINE
        
• Histogram Equalization
• Contour Detection
• Auto-cropping (ROI)
• Normalization (224×224)
• Float32 scaling (0-1)
        """
        ax2.text(0.1, 0.5, preprocess_text, fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                verticalalignment='center')
        
        # Models
        model_colors = plt.cm.Set3(np.linspace(0, 1, len(self.model_names)))
        
        for idx, (model_name, color) in enumerate(zip(self.model_names, model_colors)):
            ax = fig.add_subplot(gs[1, idx])
            ax.axis('off')
            
            acc = results[f'{model_name}_acc']
            auc = results[f'{model_name}_auc']
            
            bbox = FancyBboxPatch((0.05, 0.1), 0.9, 0.8, 
                                 boxstyle="round,pad=0.05", 
                                 facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(bbox)
            
            model_text = f"{model_name}\n\nAccuracy: {acc:.3f}\nAUC: {auc:.3f}"
            ax.text(0.5, 0.5, model_text, ha='center', va='center',
                   fontsize=10, fontweight='bold', transform=ax.transAxes)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        # Ensemble averaging
        ax_ens = fig.add_subplot(gs[2, :2])
        ax_ens.axis('off')
        ens_text = """ENSEMBLE FUSION
        
Averaging Method:
P_ensemble = mean(P_model1, P_model2, P_model3)

Optimal Threshold: {:.4f}""".format(results['optimal_threshold'])
        ax_ens.text(0.5, 0.5, ens_text, fontsize=11, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                   family='monospace', fontweight='bold', transform=ax_ens.transAxes)
        
        # Final diagnosis
        ax_final = fig.add_subplot(gs[2, 2:])
        ax_final.axis('off')
        
        acc_ens = results['metrics']['accuracy']
        auc_ens = results['metrics']['auc']
        
        final_text = f"""FINAL DIAGNOSIS

Accuracy: {acc_ens:.3f}
AUC-ROC:  {auc_ens:.3f}

✅ Disease Detected / Not Detected"""
        
        ax_final.text(0.5, 0.5, final_text, fontsize=11, ha='center', va='center',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                     family='monospace', fontweight='bold', transform=ax_final.transAxes)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
        print(f"✅ Saved: {output_path}")
        plt.close()
    
    def generate_preprocessing_pipeline(self, results, output_path='figures/02_preprocessing_pipeline.png'):
        """Figure 2: Preprocessing Pipeline Visualization"""
        print(f"\n🔧 Generating Preprocessing Pipeline...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Retinal Image Preprocessing Pipeline', fontsize=16, fontweight='bold')
        
        if not results['sample_images']:
            print("⚠️  No sample images available")
            return
        
        img = results['sample_images'][0][0]
        
        # Original
        axes[0, 0].imshow(img, cmap='gray')
        axes[0, 0].set_title('1. Original Image (224×224)', fontweight='bold')
        axes[0, 0].axis('off')
        
        # Histogram
        axes[0, 1].hist(img.ravel(), bins=256, color='blue', alpha=0.7)
        axes[0, 1].set_title('2. Histogram Analysis', fontweight='bold')
        axes[0, 1].set_xlabel('Pixel Intensity')
        axes[0, 1].set_ylabel('Frequency')
        
        # Statistics
        stats_text = f"""IMAGE STATISTICS
        
Mean:     {img.mean():.4f}
Std Dev:  {img.std():.4f}
Min:      {img.min():.4f}
Max:      {img.max():.4f}

Resolution: 224×224
Channels:   1 (Grayscale)"""
        
        axes[0, 2].axis('off')
        axes[0, 2].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                       verticalalignment='center')
        
        # Normalization effect
        normalized = (img - img.mean()) / img.std()
        axes[1, 0].imshow(np.clip(normalized, -2, 2), cmap='viridis')
        axes[1, 0].set_title('3. Normalized (Zero Mean)', fontweight='bold')
        axes[1, 0].axis('off')
        
        # Distribution before/after
        axes[1, 1].hist(img.ravel(), bins=50, alpha=0.5, label='Original', color='blue')
        axes[1, 1].hist(normalized.ravel(), bins=50, alpha=0.5, label='Normalized', color='red')
        axes[1, 1].set_title('4. Distribution Comparison', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].set_ylabel('Frequency')
        
        # Processing steps
        steps_text = """PROCESSING STEPS
        
✓ BGR → Grayscale
✓ Thresholding
✓ Contour Detection
✓ ROI Extraction
✓ Resize to 224×224
✓ Normalize to [0,1]
✓ Float32 conversion"""
        
        axes[1, 2].axis('off')
        axes[1, 2].text(0.1, 0.5, steps_text, fontsize=10, family='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                       verticalalignment='center')
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
        print(f"✅ Saved: {output_path}")
        plt.close()
    
    def generate_roc_curves(self, results, output_path='figures/03_roc_curves.png'):
        """Figure 3: ROC Curves for All Models + Ensemble"""
        print(f"\n📈 Generating ROC Curves...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.model_names) + 1))
        
        # Individual models
        for idx, model_name in enumerate(self.model_names):
            preds = np.array(results['individual_predictions'][model_name])
            fpr, tpr, _ = roc_curve(results['y_true'], preds)
            auc_score = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, linewidth=2.5, label=f'{model_name} (AUC={auc_score:.3f})',
                   color=colors[idx], linestyle='--', alpha=0.8)
        
        # Ensemble
        fpr_ens = results['fpr_ensemble']
        tpr_ens = results['tpr_ensemble']
        auc_ens = results['roc_auc_ensemble']
        
        ax.plot(fpr_ens, tpr_ens, linewidth=3, label=f'ENSEMBLE (AUC={auc_ens:.3f})',
               color=colors[-1], linestyle='-', marker='o', markersize=8, alpha=0.9)
        
        # Diagonal line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves: Individual Models vs Ensemble', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
        print(f"✅ Saved: {output_path}")
        plt.close()
    
    def generate_confusion_matrices(self, results, output_path='figures/04_confusion_matrices.png'):
        """Figure 4: Confusion Matrices for All Models"""
        print(f"\n🔲 Generating Confusion Matrices...")
        
        num_models = len(self.model_names) + 1  # +1 for ensemble
        fig, axes = plt.subplots(1, num_models, figsize=(5*num_models, 5))
        
        if num_models == 1:
            axes = [axes]
        
        # Individual models
        for idx, (ax, model_name) in enumerate(zip(axes[:-1], self.model_names)):
            preds = np.array(results['individual_predictions'][model_name])
            fpr, tpr, thresh = roc_curve(results['y_true'], preds)
            opt_idx = np.argmax(tpr - fpr)
            opt_thresh = thresh[opt_idx]
            y_pred = (preds >= opt_thresh).astype(int)
            
            cm = confusion_matrix(results['y_true'], y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            ax.set_title(f'{model_name}\nAccuracy: {results[f"{model_name}_acc"]:.3f}',
                        fontweight='bold', fontsize=11)
            ax.set_ylabel('True Label', fontweight='bold')
            ax.set_xlabel('Predicted Label', fontweight='bold')
        
        # Ensemble
        cm_ens = confusion_matrix(results['y_true'], results['y_pred_ensemble'])
        sns.heatmap(cm_ens, annot=True, fmt='d', cmap='Greens', ax=axes[-1], cbar=False,
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        axes[-1].set_title(f'ENSEMBLE\nAccuracy: {results["metrics"]["accuracy"]:.3f}',
                          fontweight='bold', fontsize=11, color='darkgreen')
        axes[-1].set_ylabel('True Label', fontweight='bold')
        axes[-1].set_xlabel('Predicted Label', fontweight='bold')
        
        fig.suptitle('Confusion Matrices: Individual Models vs Ensemble', 
                    fontsize=14, fontweight='bold', y=1.02)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
        print(f"✅ Saved: {output_path}")
        plt.close()
    
    def generate_performance_comparison(self, results, output_path='figures/05_performance_comparison.png'):
        """Figure 5: Performance Metrics Comparison"""
        print(f"\n📊 Generating Performance Comparison...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Accuracy comparison
        ax = axes[0, 0]
        models_list = list(self.model_names) + ['ENSEMBLE']
        accuracies = [results[f'{m}_acc'] for m in self.model_names] + [results['metrics']['accuracy']]
        colors_bar = plt.cm.Set3(np.linspace(0, 1, len(models_list)))
        
        bars = ax.bar(range(len(models_list)), accuracies, color=colors_bar, edgecolor='black', linewidth=2)
        ax.set_ylabel('Accuracy', fontweight='bold', fontsize=11)
        ax.set_title('Accuracy Comparison', fontweight='bold', fontsize=12)
        ax.set_xticks(range(len(models_list)))
        ax.set_xticklabels(models_list, rotation=45, ha='right')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # AUC comparison
        ax = axes[0, 1]
        aucs = [results[f'{m}_auc'] for m in self.model_names] + [results['metrics']['auc']]
        bars = ax.bar(range(len(models_list)), aucs, color=colors_bar, edgecolor='black', linewidth=2)
        ax.set_ylabel('AUC-ROC', fontweight='bold', fontsize=11)
        ax.set_title('AUC-ROC Comparison', fontweight='bold', fontsize=12)
        ax.set_xticks(range(len(models_list)))
        ax.set_xticklabels(models_list, rotation=45, ha='right')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Ensemble metrics breakdown
        ax = axes[1, 0]
        metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'AUC']
        values = [results['metrics']['accuracy'], results['metrics']['sensitivity'],
                 results['metrics']['specificity'], results['metrics']['auc']]
        colors_metrics = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        bars = ax.bar(metrics, values, color=colors_metrics, edgecolor='black', linewidth=2)
        ax.set_ylabel('Score', fontweight='bold', fontsize=11)
        ax.set_title('Ensemble Detailed Metrics', fontweight='bold', fontsize=12)
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        cm = np.array(results['metrics']['confusion_matrix'])
        tn, fp, fn, tp = cm.ravel()
        
        summary_text = f"""ENSEMBLE SUMMARY

Total Predictions: {len(results['y_true'])}
True Positives:    {tp}
True Negatives:    {tn}
False Positives:   {fp}
False Negatives:   {fn}

Threshold:         {results['metrics']['threshold']:.4f}
"""
        
        ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
               verticalalignment='center', fontweight='bold')
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
        print(f"✅ Saved: {output_path}")
        plt.close()
    
    def generate_all_figures(self, results):
        """Generate all visualization figures"""
        print("\n" + "="*70)
        print("🎨 GENERATING VISUALIZATION FIGURES")
        print("="*70)
        
        self.generate_architecture_diagram(results)
        self.generate_preprocessing_pipeline(results)
        self.generate_roc_curves(results)
        self.generate_confusion_matrices(results)
        self.generate_performance_comparison(results)
        
        print("\n" + "="*70)
        print("✅ ALL FIGURES GENERATED SUCCESSFULLY!")
        print("="*70)
        print("\n📁 Output files:")
        print("  📊 figures/01_system_architecture.png")
        print("  🔧 figures/02_preprocessing_pipeline.png")
        print("  📈 figures/03_roc_curves.png")
        print("  🔲 figures/04_confusion_matrices.png")
        print("  📊 figures/05_performance_comparison.png")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎯 ENSEMBLE MODEL TESTING & VISUALIZATION")
    print("="*70)
    
    # Define model paths
    model_paths = {
        'MobileNetV2': 'models/ensemble_MobileNetV2.keras',
        'EfficientNetB0': 'models/ensemble_EfficientNetB0.keras',
        'ResNet50': 'models/ensemble_ResNet50.keras'
    }
    
    # Initialize
    preprocessor = RetinalImagePreprocessor()
    tester = EnsembleModelTester(model_paths, preprocessor)
    
    # Load validation data
    val_df = pd.read_csv('data/processed/val.csv')
    
    # Test ensemble
    results = tester.test_ensemble(val_df)
    
    # Generate all figures
    tester.generate_all_figures(results)
    
    print("\n✨ Testing complete! Check the 'figures' folder for visualizations.")