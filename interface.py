"""
EyeMRI AI - Inference Script
Test the trained model on new retinal images (Binary Classification: DR vs No DR)
"""

import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

class RetinalImagePreprocessor:
    """Preprocesses retinal fundus images"""
    
    def __init__(self, image_size=224):
        self.image_size = image_size
        
    def crop_black_borders(self, img):
        """Remove black borders from fundus images"""
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
            
            return img[y:y+h, x:x+w]
        
        return img
    
    def apply_clahe(self, img):
        """Apply CLAHE for better contrast"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    def preprocess(self, img_path):
        """Full preprocessing pipeline"""
        img = cv2.imread(str(img_path))
        
        if img is None:
            return None
        
        img = self.crop_black_borders(img)
        img = self.apply_clahe(img)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0
        
        return img

class EyeMRIPredictor:
    """Prediction interface for EyeMRI AI (Binary Classification)"""
    
    def __init__(self, model_path='models/best_model.h5'):
        print(f"Loading model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        self.preprocessor = RetinalImagePreprocessor(image_size=224)
        
        # Binary classification
        self.class_names = ['No DR', 'DR']
        
        self.severity_info = {
            'No DR': {
                'risk': 'Low',
                'color': 'green',
                'recommendation': 'No signs of diabetic retinopathy detected. Continue regular eye checkups annually.'
            },
            'DR': {
                'risk': 'Present',
                'color': 'red',
                'recommendation': 'Signs of diabetic retinopathy detected. Please consult an ophthalmologist for detailed examination and treatment options.'
            }
        }
        
        print("✅ Model loaded successfully")
    
    def predict(self, image_path):
        """Make prediction on single image"""
        
        # Preprocess image
        img = self.preprocessor.preprocess(image_path)
        
        if img is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        # Add batch dimension
        img_batch = np.expand_dims(img, axis=0)
        
        # Predict (binary classification returns single probability)
        prediction = self.model.predict(img_batch, verbose=0)[0][0]
        
        # probability of DR (class 1)
        dr_probability = prediction * 100
        no_dr_probability = (1 - prediction) * 100
        
        # Determine predicted class
        predicted_class = 'DR' if prediction > 0.5 else 'No DR'
        confidence = max(dr_probability, no_dr_probability)
        
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'dr_probability': dr_probability,
            'no_dr_probability': no_dr_probability,
            'risk_level': self.severity_info[predicted_class]['risk'],
            'recommendation': self.severity_info[predicted_class]['recommendation']
        }
        
        return result
    
    def visualize_prediction(self, image_path, save_path=None):
        """Visualize prediction with detailed breakdown"""
        
        result = self.predict(image_path)
        
        if result is None:
            print("❌ Could not process image")
            return None
        
        # Load original image
        original = cv2.imread(str(image_path))
        if original is None:
            print(f"❌ Could not load image: {image_path}")
            return None
            
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig = plt.figure(figsize=(14, 6))
        
        # Original image
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(original)
        ax1.set_title('Retinal Fundus Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Prediction details
        ax2 = plt.subplot(1, 2, 2)
        ax2.axis('off')
        
        # Title
        severity_color = self.severity_info[result['predicted_class']]['color']
        title_text = f"Prediction: {result['predicted_class']}"
        ax2.text(0.5, 0.95, title_text, 
                ha='center', va='top', fontsize=18, fontweight='bold',
                color=severity_color, transform=ax2.transAxes)
        
        # Confidence
        confidence_text = f"Confidence: {result['confidence']:.1f}%"
        ax2.text(0.5, 0.86, confidence_text,
                ha='center', va='top', fontsize=14,
                transform=ax2.transAxes)
        
        # Risk level
        risk_text = f"Risk Level: {result['risk_level']}"
        ax2.text(0.5, 0.78, risk_text,
                ha='center', va='top', fontsize=12,
                bbox=dict(boxstyle='round', facecolor=severity_color, alpha=0.3),
                transform=ax2.transAxes)
        
        # Probability bar chart
        classes = ['No DR', 'DR']
        values = [result['no_dr_probability'], result['dr_probability']]
        colors = ['green', 'red']
        
        bars_ax = fig.add_axes([0.58, 0.40, 0.35, 0.25])
        bars = bars_ax.barh(classes, values, color=colors, alpha=0.6)
        
        # Highlight predicted class
        predicted_idx = 0 if result['predicted_class'] == 'No DR' else 1
        bars[predicted_idx].set_alpha(1.0)
        bars[predicted_idx].set_edgecolor('black')
        bars[predicted_idx].set_linewidth(2)
        
        bars_ax.set_xlabel('Probability (%)', fontsize=10)
        bars_ax.set_title('Classification Probabilities', fontsize=12, fontweight='bold')
        bars_ax.set_xlim(0, 100)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            bars_ax.text(val + 1, i, f'{val:.1f}%', 
                        va='center', fontsize=10, fontweight='bold')
        
        # Recommendation box
        rec_text = f"Recommendation:\n{result['recommendation']}"
        ax2.text(0.5, 0.22, rec_text,
                ha='center', va='top', fontsize=9,
                wrap=True, transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # Disclaimer
        disclaimer = "⚠️ This is NOT a medical diagnosis.\nPlease consult a qualified ophthalmologist."
        ax2.text(0.5, 0.02, disclaimer,
                ha='center', va='bottom', fontsize=8,
                style='italic', color='red', fontweight='bold',
                transform=ax2.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return result
    
    def batch_predict(self, image_dir, output_csv='predictions.csv'):
        """Predict on multiple images"""
        
        import pandas as pd
        
        image_dir = Path(image_dir)
        
        # Find all image files
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            image_files.extend(list(image_dir.glob(ext)))
        
        print(f"Found {len(image_files)} images in {image_dir}")
        
        results = []
        
        for img_path in image_files:
            print(f"Processing: {img_path.name}")
            result = self.predict(img_path)
            
            if result:
                results.append({
                    'filename': img_path.name,
                    'predicted_class': result['predicted_class'],
                    'confidence': result['confidence'],
                    'dr_probability': result['dr_probability'],
                    'risk_level': result['risk_level']
                })
        
        if results:
            df = pd.DataFrame(results)
            df.to_csv(output_csv, index=False)
            
            print(f"\n✅ Predictions saved to: {output_csv}")
            print(f"\n📊 Summary:")
            print(df['predicted_class'].value_counts())
            print(f"\nAverage DR probability: {df['dr_probability'].mean():.2f}%")
            
            return df
        else:
            print("❌ No valid predictions made")
            return None

def test_on_sample_images():
    """Test model on sample validation images"""
    
    import pandas as pd
    
    # Load validation data
    val_csv = Path('data/processed/val.csv')
    if not val_csv.exists():
        print(f"❌ Validation CSV not found: {val_csv}")
        return
    
    val_df = pd.read_csv(val_csv)
    
    # Initialize predictor
    predictor = EyeMRIPredictor()
    
    # Test on random samples from each class
    samples = []
    
    # Get samples from each class
    for diagnosis in [0, 1]:  # Binary: 0=No DR, 1=DR
        class_samples = val_df[val_df['diagnosis'] == diagnosis]
        if len(class_samples) > 0:
            # Get up to 3 samples from each class
            n_samples = min(3, len(class_samples))
            samples.extend(class_samples.sample(n_samples).to_dict('records'))
    
    print("\n" + "="*60)
    print(f"Testing Model on {len(samples)} Sample Images")
    print("="*60)
    
    results_dir = Path('test_results')
    results_dir.mkdir(exist_ok=True)
    
    correct = 0
    total = 0
    
    for i, sample in enumerate(samples, 1):
        img_path = Path(sample['image_path'])
        true_class = 'DR' if sample['diagnosis'] == 1 else 'No DR'
        
        if not img_path.exists():
            print(f"\n⚠️ Sample {i}: Image not found: {img_path}")
            continue
        
        print(f"\n--- Sample {i} ---")
        print(f"Image: {img_path.name}")
        print(f"True class: {true_class}")
        
        result = predictor.visualize_prediction(
            img_path,
            save_path=results_dir / f'test_sample_{i}.png'
        )
        
        if result:
            print(f"Predicted: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.1f}%")
            
            is_correct = result['predicted_class'] == true_class
            if is_correct:
                print("✅ Correct prediction")
                correct += 1
            else:
                print("❌ Incorrect prediction")
            
            total += 1
    
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\n{'='*60}")
        print(f"Sample Test Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        print(f"{'='*60}")

if __name__ == "__main__":
    print("="*60)
    print("EyeMRI AI - Model Testing")
    print("="*60)
    
    # Run tests
    test_on_sample_images()
    
    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60)
    print("\n📖 Usage Examples:")
    print("\n1. Test single image:")
    print("   predictor = EyeMRIPredictor()")
    print("   result = predictor.visualize_prediction('path/to/image.png')")
    print("\n2. Batch predict on folder:")
    print("   predictor.batch_predict('path/to/images/', 'output.csv')")
    print("\n3. Get prediction only (no visualization):")
    print("   result = predictor.predict('path/to/image.png')")
    print("   print(result)")