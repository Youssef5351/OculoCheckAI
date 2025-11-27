# test_models.py
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import efficientnet
import cv2
from PIL import Image
import numpy as np
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image exactly like the Flask app"""
    try:
        # Load image
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Resize
        img_array = cv2.resize(img_array, target_size)
        
        # Apply preprocessing (same as Flask app)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            padding = 5
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img_array.shape[1] - x, w + 2 * padding)
            h = min(img_array.shape[0] - y, h + 2 * padding)
            img_array = img_array[y:y+h, x:x+w]
            img_array = cv2.resize(img_array, target_size)
        
        # Normalize and preprocess for EfficientNet
        img_array = img_array.astype(np.float32) / 255.0
        img_array = efficientnet.preprocess_input(img_array * 255.0)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
        
    except Exception as e:
        logger.error(f"Error preprocessing {image_path}: {e}")
        return None

def test_single_model(model_path, test_images, model_name):
    """Test a single model with multiple test images"""
    print(f"\n{'='*60}")
    print(f"TESTING {model_name.upper()} MODEL")
    print(f"{'='*60}")
    
    try:
        # Load model
        print(f"Loading model from: {model_path}")
        model = load_model(model_path)
        print("✅ Model loaded successfully")
        
        # Test each image
        for image_path, expected_label in test_images:
            print(f"\n--- Testing: {os.path.basename(image_path)} ---")
            print(f"Expected: {expected_label}")
            
            # Preprocess image
            processed_image = preprocess_image(image_path)
            if processed_image is None:
                print("❌ Failed to preprocess image")
                continue
            
            # Make prediction
            prediction = model.predict(processed_image, verbose=0)
            
            # Interpret results based on model output shape
            if len(prediction[0]) == 1:
                # Binary classification
                confidence = float(prediction[0][0])
                if confidence >= 0.5:
                    predicted_class = "Disease"
                    confidence_score = confidence
                else:
                    predicted_class = "Normal"
                    confidence_score = 1 - confidence
            else:
                # Multi-class classification
                predicted_idx = np.argmax(prediction[0])
                confidence_score = float(prediction[0][predicted_idx])
                predicted_class = f"Class_{predicted_idx}"
            
            print(f"Predicted: {predicted_class}")
            print(f"Confidence: {confidence_score:.4f}")
            print(f"Raw prediction: {prediction[0]}")
            
            # Check if prediction matches expected
            is_correct = (predicted_class.lower() in expected_label.lower() or 
                         expected_label.lower() in predicted_class.lower())
            print(f"✅ CORRECT" if is_correct else "❌ WRONG")
            
    except Exception as e:
        print(f"❌ Error testing {model_name}: {e}")
        return False
    
    return True

def test_all_models():
    """Test all models with sample images"""
    
    # Define test images for each disease
    # Replace these paths with your actual test images
    test_images = {
        'diabetes': [
            ('test_images/diabetes_positive.jpg', 'disease'),
            ('test_images/diabetes_negative.jpg', 'normal'),
        ],
        'amd': [
            ('test_images/amd_positive.jpg', 'disease'),
            ('test_images/amd_negative.jpg', 'normal'),
        ],
        'hypertension': [
            ('test_images/hypertension_positive.jpg', 'disease'),
            ('test_images/hypertension_negative.jpg', 'normal'),
        ],
        'cataract': [
            ('test_images/cataract_positive.jpg', 'disease'),
            ('test_images/cataract_negative.jpg', 'normal'),
        ],
        'glaucoma': [
            ('test_images/glaucoma_positive.jpg', 'disease'),
            ('test_images/glaucoma_negative.jpg', 'normal'),
        ]
    }
    
    # Model paths
    model_paths = {
        'diabetes': './models/ensemble_EfficientNetB0.keras',
        'amd': './models/amd_classifier_final_improved.keras',
        'hypertension': './models/hypertension_best.keras',
        'cataract': './models/cataract_best.keras',
        'glaucoma': './models/glaucoma_EfficientNetB0.keras'
    }
    
    # Check which test images actually exist
    print("Checking test images...")
    for disease, images in test_images.items():
        existing_images = []
        for img_path, label in images:
            if os.path.exists(img_path):
                existing_images.append((img_path, label))
            else:
                print(f"⚠️  Test image not found: {img_path}")
        
        if existing_images:
            test_images[disease] = existing_images
        else:
            print(f"❌ No test images found for {disease}")
    
    # Test each model
    results = {}
    for disease, model_path in model_paths.items():
        if os.path.exists(model_path) and disease in test_images and test_images[disease]:
            success = test_single_model(model_path, test_images[disease], disease)
            results[disease] = success
        else:
            print(f"\n❌ Skipping {disease} - model or test images not found")
            results[disease] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    for disease, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{disease:15} {status}")

def quick_model_inspection():
    """Quickly inspect each model's architecture and input/output specs"""
    print(f"\n{'='*60}")
    print("MODEL INSPECTION")
    print(f"{'='*60}")
    
    model_paths = {
        'diabetes': './models/ensemble_EfficientNetB0.keras',
        'amd': './models/amd_classifier_final_improved.keras',
        'hypertension': './models/hypertension_best.keras',
        'cataract': './models/cataract_best.keras',
        'glaucoma': './models/glaucoma_EfficientNetB0.keras'
    }
    
    for disease, model_path in model_paths.items():
        print(f"\n🔍 Inspecting {disease} model...")
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            continue
            
        try:
            model = load_model(model_path)
            
            print(f"✅ Model loaded successfully")
            print(f"   Input shape: {model.input_shape}")
            print(f"   Output shape: {model.output_shape}")
            print(f"   Number of layers: {len(model.layers)}")
            print(f"   Model type: {type(model)}")
            
            # Check first and last layers
            if model.layers:
                print(f"   First layer: {model.layers[0].__class__.__name__}")
                print(f"   Last layer: {model.layers[-1].__class__.__name__}")
            
            # Check if it's a binary or multi-class classifier
            output_shape = model.output_shape
            if len(output_shape) == 2:
                if output_shape[1] == 1:
                    print(f"   ⚡ Binary classifier (sigmoid output)")
                else:
                    print(f"   ⚡ Multi-class classifier ({output_shape[1]} classes)")
            
        except Exception as e:
            print(f"❌ Error inspecting {disease}: {e}")

def test_preprocessing():
    """Test if preprocessing is working correctly"""
    print(f"\n{'='*60}")
    print("PREPROCESSING TEST")
    print(f"{'='*60}")
    
    # Find any image to test preprocessing
    test_image = None
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_image = os.path.join(root, file)
                break
        if test_image:
            break
    
    if test_image:
        print(f"Testing preprocessing with: {test_image}")
        processed = preprocess_image(test_image)
        if processed is not None:
            print(f"✅ Preprocessing successful")
            print(f"   Input shape: {processed.shape}")
            print(f"   Data type: {processed.dtype}")
            print(f"   Value range: [{processed.min():.3f}, {processed.max():.3f}]")
        else:
            print("❌ Preprocessing failed")
    else:
        print("⚠️  No test image found for preprocessing test")

if __name__ == '__main__':
    print("🧪 RETINAL DISEASE MODEL TEST SUITE")
    print("This will test each model with sample images and inspect their architecture")
    
    # First, inspect models
    quick_model_inspection()
    
    # Test preprocessing
    test_preprocessing()
    
    # Test all models with actual images
    test_all_models()
    
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print("1. Check if model input/output shapes match our preprocessing")
    print("2. Verify test images are in the correct format")
    print("3. Check if models were trained with similar preprocessing")
    print("4. Look for any normalization differences")
    print(f"{'='*60}")