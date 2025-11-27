from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import numpy as np
import io
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ============================================================================
# LOAD MODELS INDEPENDENTLY
# ============================================================================

MODELS = {
    'diabetic_retinopathy': None,
    'cataract': None,
    'glaucoma': None,
    'hypertension': None,
    'amd': None
}

# Load Diabetic Retinopathy Model
try:
    dr_model_path = './models/ensemble_EfficientNetB0.keras'
    if os.path.exists(dr_model_path):
        MODELS['diabetic_retinopathy'] = load_model(dr_model_path)
        logger.info("✅ Diabetic Retinopathy model loaded successfully")
    else:
        logger.warning(f"⚠️  DR model not found at {dr_model_path}")
except Exception as e:
    logger.error(f"❌ Error loading DR model: {e}")

# Load Cataract Model
try:
    cataract_model_path = './models/cataract_best.keras'
    if os.path.exists(cataract_model_path):
        MODELS['cataract'] = load_model(cataract_model_path)
        logger.info("✅ Cataract model loaded successfully")
    else:
        logger.warning(f"⚠️  Cataract model not found at {cataract_model_path}")
except Exception as e:
    logger.error(f"❌ Error loading Cataract model: {e}")

# Load Glaucoma Model
try:
    glaucoma_model_path = './models/glaucoma_EfficientNetB0.keras'
    if os.path.exists(glaucoma_model_path):
        MODELS['glaucoma'] = load_model(glaucoma_model_path)
        logger.info("✅ Glaucoma model loaded successfully")
    else:
        logger.warning(f"⚠️  Glaucoma model not found at {glaucoma_model_path}")
except Exception as e:
    logger.error(f"❌ Error loading Glaucoma model: {e}")

# Load Hypertension Model
try:
    hypertension_model_path = './models/hypertension_best.keras'
    if os.path.exists(hypertension_model_path):
        MODELS['hypertension'] = load_model(hypertension_model_path)
        logger.info("✅ Hypertension model loaded successfully")
    else:
        logger.warning(f"⚠️  Hypertension model not found at {hypertension_model_path}")
except Exception as e:
    logger.error(f"❌ Error loading Hypertension model: {e}")

# Load AMD Model
try:
    amd_model_path = './models/amd_classifier_final_improved.keras'
    if os.path.exists(amd_model_path):
        MODELS['amd'] = load_model(amd_model_path)
        logger.info("✅ AMD model loaded successfully")
    else:
        logger.warning(f"⚠️  AMD model not found at {amd_model_path}")
except Exception as e:
    logger.error(f"❌ Error loading AMD model: {e}")

# ============================================================================
# PREPROCESSING FUNCTIONS - SEPARATE FOR EACH DISEASE
# ============================================================================

def preprocess_for_diabetic_retinopathy(image):
    """
    Preprocessing specifically for DR model
    Uses padding=5 as in DR training
    """
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Crop with padding=5 (DR specific)
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            padding = 5  # DR uses padding=5
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img_array.shape[1] - x, w + 2 * padding)
            h = min(img_array.shape[0] - y, h + 2 * padding)
            img_array = img_array[y:y+h, x:x+w]
        
        img_array = cv2.resize(img_array, (224, 224))
        img_array = img_array.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        logger.error(f"Error in DR preprocessing: {e}")
        raise e

def preprocess_for_cataract(image):
    """
    Preprocessing specifically for Cataract model
    Uses padding=10 as in Cataract training
    """
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Crop with padding=10 (Cataract specific)
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            padding = 10  # Cataract uses padding=10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img_array.shape[1] - x, w + 2 * padding)
            h = min(img_array.shape[0] - y, h + 2 * padding)
            img_array = img_array[y:y+h, x:x+w]
        
        img_array = cv2.resize(img_array, (224, 224))
        img_array = img_array.astype(np.float32) / 255.0  # FIXED: Changed 'ast' to 'astype'
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        logger.error(f"Error in Cataract preprocessing: {e}")
        raise e

def preprocess_for_glaucoma(image):
    """
    Preprocessing specifically for Glaucoma model
    Uses CLAHE enhancement for better vessel visibility
    """
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Crop with padding=10
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img_array.shape[1] - x, w + 2 * padding)
            h = min(img_array.shape[0] - y, h + 2 * padding)
            img_array = img_array[y:y+h, x:x+w]
        
        # CLAHE enhancement for glaucoma detection
        lab = cv2.cvtColor(img_array, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        img_array = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        img_array = cv2.resize(img_array, (224, 224))
        img_array = img_array.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        logger.error(f"Error in Glaucoma preprocessing: {e}")
        raise e

def preprocess_for_hypertension(image):
    """
    Preprocessing specifically for Hypertension model
    Uses enhanced preprocessing with CLAHE for better vessel visibility
    """
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Enhanced preprocessing for hypertension features
        # Focus on blood vessels and optic disc
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest reasonable contour
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > img_array.shape[1] * 0.3 and h > img_array.shape[0] * 0.3:
                    # Add padding
                    pad_x = max(10, int(w * 0.02))
                    pad_y = max(10, int(h * 0.02))
                    x = max(0, x - pad_x)
                    y = max(0, y - pad_y)
                    w = min(img_array.shape[1] - x, w + 2 * pad_x)
                    h = min(img_array.shape[0] - y, h + 2 * pad_y)
                    img_array = img_array[y:y+h, x:x+w]
                    break
        
        # CLAHE enhancement for better vessel visibility
        lab = cv2.cvtColor(img_array, cv2.COLOR_BGR2LAB)
        lab_planes = list(cv2.split(lab))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Resize
        img_array = cv2.resize(img_array, (224, 224))
        
        # Normalize
        img_array = img_array.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        logger.error(f"Error in Hypertension preprocessing: {e}")
        raise e

def preprocess_for_amd(image):
    """
    Preprocessing specifically for AMD model
    """
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Crop with padding
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            padding = 15
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img_array.shape[1] - x, w + 2 * padding)
            h = min(img_array.shape[0] - y, h + 2 * padding)
            img_array = img_array[y:y+h, x:x+w]
        
        img_array = cv2.resize(img_array, (224, 224))
        img_array = img_array.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        logger.error(f"Error in AMD preprocessing: {e}")
        raise e

def validate_image(file):
    """Validate the uploaded image file"""
    try:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > 20 * 1024 * 1024:
            return False, "File size too large. Maximum size is 20MB."
        
        try:
            image = Image.open(file)
            image.verify()
            file.seek(0)
            return True, "Valid"
        except Exception:
            return False, "Invalid image file"
            
    except Exception as e:
        return False, f"Validation error: {str(e)}"

# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy" if any(MODELS.values()) else "unhealthy",
        "models": {
            "diabetic_retinopathy": MODELS['diabetic_retinopathy'] is not None,
            "cataract": MODELS['cataract'] is not None,
            "glaucoma": MODELS['glaucoma'] is not None,
            "hypertension": MODELS['hypertension'] is not None,
            "amd": MODELS['amd'] is not None
        }
    })

# ============================================================================
# DIABETIC RETINOPATHY - COMPLETELY ISOLATED
# ============================================================================

@app.route('/api/diabetic-retinopathy/predict', methods=['POST'])
def predict_diabetic_retinopathy():
    """
    ISOLATED endpoint for Diabetic Retinopathy detection only
    """
    try:
        if MODELS['diabetic_retinopathy'] is None:
            return jsonify({
                'error': 'Diabetic Retinopathy model not loaded',
                'success': False
            }), 503

        if 'file' not in request.files:
            return jsonify({
                'error': 'No file uploaded',
                'success': False
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'success': False
            }), 400

        # Validate
        is_valid, validation_msg = validate_image(file)
        if not is_valid:
            return jsonify({
                'error': validation_msg,
                'success': False
            }), 400

        # Preprocess using DR-specific function
        try:
            image = Image.open(io.BytesIO(file.read()))
            processed_image = preprocess_for_diabetic_retinopathy(image)
        except Exception as e:
            return jsonify({
                'error': f'Error processing image: {str(e)}',
                'success': False
            }), 400
        
        # Predict using DR model ONLY
        try:
            model = MODELS['diabetic_retinopathy']
            predictions = model.predict(processed_image, verbose=0)
            
            dr_confidence = float(predictions[0][0])
            normal_confidence = 1.0 - dr_confidence
            
            if dr_confidence >= 0.5:
                result = "Diabetic Retinopathy Detected"
                confidence = dr_confidence
                has_disease = True
            else:
                result = "No Diabetic Retinopathy"
                confidence = normal_confidence
                has_disease = False
            
            logger.info(f"DR Prediction: {result} ({confidence:.4f})")
            
            return jsonify({
                'result': result,
                'confidence': confidence,
                'has_disease': has_disease,
                'disease_confidence': dr_confidence,
                'normal_confidence': normal_confidence,
                'success': True
            })
            
        except Exception as e:
            logger.error(f"DR Prediction error: {e}")
            return jsonify({
                'error': f'Prediction failed: {str(e)}',
                'success': False
            }), 500
        
    except Exception as e:
        logger.error(f"Unexpected error in DR prediction: {e}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'success': False
        }), 500

@app.route('/api/diabetic-retinopathy/batch-predict', methods=['POST'])
def batch_predict_diabetic_retinopathy():
    """
    ISOLATED batch prediction for Diabetic Retinopathy only
    """
    try:
        if MODELS['diabetic_retinopathy'] is None:
            return jsonify({
                'error': 'Diabetic Retinopathy model not loaded',
                'success': False
            }), 503

        files = request.files.getlist('files')
        
        if not files or len(files) == 0:
            return jsonify({
                'error': 'No files uploaded',
                'success': False
            }), 400
        
        results = []
        model = MODELS['diabetic_retinopathy']
        
        for file in files:
            try:
                if file.filename == '':
                    continue
                
                is_valid, validation_msg = validate_image(file)
                if not is_valid:
                    results.append({
                        'filename': file.filename,
                        'result': 'Invalid File',
                        'confidence': 0,
                        'error': validation_msg,
                        'success': False
                    })
                    continue

                image = Image.open(io.BytesIO(file.read()))
                processed_image = preprocess_for_diabetic_retinopathy(image)
                
                predictions = model.predict(processed_image, verbose=0)
                dr_confidence = float(predictions[0][0])
                normal_confidence = 1.0 - dr_confidence

                if dr_confidence >= 0.5:
                    result = "Diabetic Retinopathy Detected"
                    confidence = dr_confidence
                    has_disease = True
                else:
                    result = "No Diabetic Retinopathy"
                    confidence = normal_confidence
                    has_disease = False
                
                results.append({
                    'filename': file.filename,
                    'result': result,
                    'confidence': confidence,
                    'has_disease': has_disease,
                    'disease_confidence': dr_confidence,
                    'normal_confidence': normal_confidence,
                    'success': True
                })
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {e}")
                results.append({
                    'filename': file.filename,
                    'result': 'Processing Error',
                    'confidence': 0,
                    'error': str(e),
                    'success': False
                })
        
        return jsonify({
            'results': results,
            'success': True,
            'total_processed': len(results),
            'successful': len([r for r in results if r.get('success', False)])
        })
        
    except Exception as e:
        logger.error(f"Unexpected error in DR batch prediction: {e}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'success': False
        }), 500

# ============================================================================
# CATARACT - COMPLETELY ISOLATED
# ============================================================================

@app.route('/api/cataract/predict', methods=['POST'])
def predict_cataract():
    """
    ISOLATED endpoint for Cataract detection only
    """
    try:
        if MODELS['cataract'] is None:
            return jsonify({
                'error': 'Cataract model not loaded',
                'success': False
            }), 503

        if 'file' not in request.files:
            return jsonify({
                'error': 'No file uploaded',
                'success': False
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'success': False
            }), 400

        # Validate
        is_valid, validation_msg = validate_image(file)
        if not is_valid:
            return jsonify({
                'error': validation_msg,
                'success': False
            }), 400

        # Preprocess using Cataract-specific function
        try:
            image = Image.open(io.BytesIO(file.read()))
            processed_image = preprocess_for_cataract(image)
        except Exception as e:
            return jsonify({
                'error': f'Error processing image: {str(e)}',
                'success': False
            }), 400
        
        # Predict using Cataract model ONLY
        try:
            model = MODELS['cataract']
            predictions = model.predict(processed_image, verbose=0)
            
            cataract_confidence = float(predictions[0][0])
            normal_confidence = 1.0 - cataract_confidence
            
            if cataract_confidence >= 0.5:
                result = "Cataract Detected"
                confidence = cataract_confidence
                has_disease = True
            else:
                result = "No Cataract"
                confidence = normal_confidence
                has_disease = False
            
            logger.info(f"Cataract Prediction: {result} ({confidence:.4f})")
            
            return jsonify({
                'result': result,
                'confidence': confidence,
                'has_disease': has_disease,
                'disease_confidence': cataract_confidence,
                'normal_confidence': normal_confidence,
                'success': True
            })
            
        except Exception as e:
            logger.error(f"Cataract Prediction error: {e}")
            return jsonify({
                'error': f'Prediction failed: {str(e)}',
                'success': False
            }), 500
        
    except Exception as e:
        logger.error(f"Unexpected error in Cataract prediction: {e}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'success': False
        }), 500

@app.route('/api/cataract/batch-predict', methods=['POST'])
def batch_predict_cataract():
    """
    ISOLATED batch prediction for Cataract only
    """
    try:
        if MODELS['cataract'] is None:
            return jsonify({
                'error': 'Cataract model not loaded',
                'success': False
            }), 503

        files = request.files.getlist('files')
        
        if not files or len(files) == 0:
            return jsonify({
                'error': 'No files uploaded',
                'success': False
            }), 400
        
        results = []
        model = MODELS['cataract']
        
        for file in files:
            try:
                if file.filename == '':
                    continue
                
                is_valid, validation_msg = validate_image(file)
                if not is_valid:
                    results.append({
                        'filename': file.filename,
                        'result': 'Invalid File',
                        'confidence': 0,
                        'error': validation_msg,
                        'success': False
                    })
                    continue

                image = Image.open(io.BytesIO(file.read()))
                processed_image = preprocess_for_cataract(image)
                
                predictions = model.predict(processed_image, verbose=0)
                cataract_confidence = float(predictions[0][0])
                normal_confidence = 1.0 - cataract_confidence

                if cataract_confidence >= 0.5:
                    result = "Cataract Detected"
                    confidence = cataract_confidence
                    has_disease = True
                else:
                    result = "No Cataract"
                    confidence = normal_confidence
                    has_disease = False
                
                results.append({
                    'filename': file.filename,
                    'result': result,
                    'confidence': confidence,
                    'has_disease': has_disease,
                    'disease_confidence': cataract_confidence,
                    'normal_confidence': normal_confidence,
                    'success': True
                })
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {e}")
                results.append({
                    'filename': file.filename,
                    'result': 'Processing Error',
                    'confidence': 0,
                    'error': str(e),
                    'success': False
                })
        
        return jsonify({
            'results': results,
            'success': True,
            'total_processed': len(results),
            'successful': len([r for r in results if r.get('success', False)])
        })
        
    except Exception as e:
        logger.error(f"Unexpected error in Cataract batch prediction: {e}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'success': False
        }), 500

# ============================================================================
# GLAUCOMA - COMPLETELY ISOLATED
# ============================================================================

@app.route('/api/glaucoma/predict', methods=['POST'])
def predict_glaucoma():
    """
    ISOLATED endpoint for Glaucoma detection only
    """
    try:
        if MODELS['glaucoma'] is None:
            return jsonify({
                'error': 'Glaucoma model not loaded',
                'success': False
            }), 503

        if 'file' not in request.files:
            return jsonify({
                'error': 'No file uploaded',
                'success': False
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'success': False
            }), 400

        # Validate
        is_valid, validation_msg = validate_image(file)
        if not is_valid:
            return jsonify({
                'error': validation_msg,
                'success': False
            }), 400

        # Preprocess using Glaucoma-specific function
        try:
            image = Image.open(io.BytesIO(file.read()))
            processed_image = preprocess_for_glaucoma(image)
        except Exception as e:
            return jsonify({
                'error': f'Error processing image: {str(e)}',
                'success': False
            }), 400
        
        # Predict using Glaucoma model ONLY
        try:
            model = MODELS['glaucoma']
            predictions = model.predict(processed_image, verbose=0)
            
            glaucoma_confidence = float(predictions[0][0])
            normal_confidence = 1.0 - glaucoma_confidence
            
            if glaucoma_confidence >= 0.5:
                result = "Glaucoma Detected"
                confidence = glaucoma_confidence
                has_disease = True
            else:
                result = "No Glaucoma"
                confidence = normal_confidence
                has_disease = False
            
            logger.info(f"Glaucoma Prediction: {result} ({confidence:.4f})")
            
            return jsonify({
                'result': result,
                'confidence': confidence,
                'has_disease': has_disease,
                'disease_confidence': glaucoma_confidence,
                'normal_confidence': normal_confidence,
                'success': True
            })
            
        except Exception as e:
            logger.error(f"Glaucoma Prediction error: {e}")
            return jsonify({
                'error': f'Prediction failed: {str(e)}',
                'success': False
            }), 500
        
    except Exception as e:
        logger.error(f"Unexpected error in Glaucoma prediction: {e}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'success': False
        }), 500

@app.route('/api/glaucoma/batch-predict', methods=['POST'])
def batch_predict_glaucoma():
    """
    ISOLATED batch prediction for Glaucoma only
    """
    try:
        if MODELS['glaucoma'] is None:
            return jsonify({
                'error': 'Glaucoma model not loaded',
                'success': False
            }), 503

        files = request.files.getlist('files')
        
        if not files or len(files) == 0:
            return jsonify({
                'error': 'No files uploaded',
                'success': False
            }), 400
        
        results = []
        model = MODELS['glaucoma']
        
        for file in files:
            try:
                if file.filename == '':
                    continue
                
                is_valid, validation_msg = validate_image(file)
                if not is_valid:
                    results.append({
                        'filename': file.filename,
                        'result': 'Invalid File',
                        'confidence': 0,
                        'error': validation_msg,
                        'success': False
                    })
                    continue

                image = Image.open(io.BytesIO(file.read()))
                processed_image = preprocess_for_glaucoma(image)
                
                predictions = model.predict(processed_image, verbose=0)
                glaucoma_confidence = float(predictions[0][0])
                normal_confidence = 1.0 - glaucoma_confidence

                if glaucoma_confidence >= 0.5:
                    result = "Glaucoma Detected"
                    confidence = glaucoma_confidence
                    has_disease = True
                else:
                    result = "No Glaucoma"
                    confidence = normal_confidence
                    has_disease = False
                
                results.append({
                    'filename': file.filename,
                    'result': result,
                    'confidence': confidence,
                    'has_disease': has_disease,
                    'disease_confidence': glaucoma_confidence,
                    'normal_confidence': normal_confidence,
                    'success': True
                })
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {e}")
                results.append({
                    'filename': file.filename,
                    'result': 'Processing Error',
                    'confidence': 0,
                    'error': str(e),
                    'success': False
                })
        
        return jsonify({
            'results': results,
            'success': True,
            'total_processed': len(results),
            'successful': len([r for r in results if r.get('success', False)])
        })
        
    except Exception as e:
        logger.error(f"Unexpected error in Glaucoma batch prediction: {e}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'success': False
        }), 500

# ============================================================================
# HYPERTENSION - COMPLETELY ISOLATED
# ============================================================================

@app.route('/api/hypertension/predict', methods=['POST'])
def predict_hypertension():
    """
    ISOLATED endpoint for Hypertension detection only
    """
    try:
        if MODELS['hypertension'] is None:
            return jsonify({
                'error': 'Hypertension model not loaded',
                'success': False
            }), 503

        if 'file' not in request.files:
            return jsonify({
                'error': 'No file uploaded',
                'success': False
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'success': False
            }), 400

        # Validate
        is_valid, validation_msg = validate_image(file)
        if not is_valid:
            return jsonify({
                'error': validation_msg,
                'success': False
            }), 400

        # Preprocess using Hypertension-specific function
        try:
            image = Image.open(io.BytesIO(file.read()))
            processed_image = preprocess_for_hypertension(image)
        except Exception as e:
            return jsonify({
                'error': f'Error processing image: {str(e)}',
                'success': False
            }), 400
        
        # Predict using Hypertension model ONLY
        try:
            model = MODELS['hypertension']
            predictions = model.predict(processed_image, verbose=0)
            
            hypertension_confidence = float(predictions[0][0])
            normal_confidence = 1.0 - hypertension_confidence
            
            if hypertension_confidence >= 0.5:
                result = "Hypertension Detected"
                confidence = hypertension_confidence
                has_disease = True
            else:
                result = "No Hypertension"
                confidence = normal_confidence
                has_disease = False
            
            logger.info(f"Hypertension Prediction: {result} ({confidence:.4f})")
            
            return jsonify({
                'result': result,
                'confidence': confidence,
                'has_disease': has_disease,
                'disease_confidence': hypertension_confidence,
                'normal_confidence': normal_confidence,
                'success': True
            })
            
        except Exception as e:
            logger.error(f"Hypertension Prediction error: {e}")
            return jsonify({
                'error': f'Prediction failed: {str(e)}',
                'success': False
            }), 500
        
    except Exception as e:
        logger.error(f"Unexpected error in Hypertension prediction: {e}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'success': False
        }), 500

@app.route('/api/hypertension/batch-predict', methods=['POST'])
def batch_predict_hypertension():
    """
    ISOLATED batch prediction for Hypertension only
    """
    try:
        if MODELS['hypertension'] is None:
            return jsonify({
                'error': 'Hypertension model not loaded',
                'success': False
            }), 503

        files = request.files.getlist('files')
        
        if not files or len(files) == 0:
            return jsonify({
                'error': 'No files uploaded',
                'success': False
            }), 400
        
        results = []
        model = MODELS['hypertension']
        
        for file in files:
            try:
                if file.filename == '':
                    continue
                
                is_valid, validation_msg = validate_image(file)
                if not is_valid:
                    results.append({
                        'filename': file.filename,
                        'result': 'Invalid File',
                        'confidence': 0,
                        'error': validation_msg,
                        'success': False
                    })
                    continue

                image = Image.open(io.BytesIO(file.read()))
                processed_image = preprocess_for_hypertension(image)
                
                predictions = model.predict(processed_image, verbose=0)
                hypertension_confidence = float(predictions[0][0])
                normal_confidence = 1.0 - hypertension_confidence

                if hypertension_confidence >= 0.5:
                    result = "Hypertension Detected"
                    confidence = hypertension_confidence
                    has_disease = True
                else:
                    result = "No Hypertension"
                    confidence = normal_confidence
                    has_disease = False
                
                results.append({
                    'filename': file.filename,
                    'result': result,
                    'confidence': confidence,
                    'has_disease': has_disease,
                    'disease_confidence': hypertension_confidence,
                    'normal_confidence': normal_confidence,
                    'success': True
                })
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {e}")
                results.append({
                    'filename': file.filename,
                    'result': 'Processing Error',
                    'confidence': 0,
                    'error': str(e),
                    'success': False
                })
        
        return jsonify({
            'results': results,
            'success': True,
            'total_processed': len(results),
            'successful': len([r for r in results if r.get('success', False)])
        })
        
    except Exception as e:
        logger.error(f"Unexpected error in Hypertension batch prediction: {e}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'success': False
        }), 500

# ============================================================================
# AMD - COMPLETELY ISOLATED
# ============================================================================

# In the AMD prediction endpoint, update the prediction logic:
@app.route('/api/amd/predict', methods=['POST'])
def predict_amd():
    """
    ISOLATED endpoint for AMD detection only
    """
    try:
        if MODELS['amd'] is None:
            return jsonify({
                'error': 'AMD model not loaded',
                'success': False
            }), 503

        if 'file' not in request.files:
            return jsonify({
                'error': 'No file uploaded',
                'success': False
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'success': False
            }), 400

        # Validate
        is_valid, validation_msg = validate_image(file)
        if not is_valid:
            return jsonify({
                'error': validation_msg,
                'success': False
            }), 400

        # Preprocess using AMD-specific function
        try:
            image = Image.open(io.BytesIO(file.read()))
            processed_image = preprocess_for_amd(image)
        except Exception as e:
            return jsonify({
                'error': f'Error processing image: {str(e)}',
                'success': False
            }), 400
        
        # Predict using AMD model ONLY
        try:
            model = MODELS['amd']
            predictions = model.predict(processed_image, verbose=0)
            
            # Debug: Print raw prediction to understand the output
            print(f"AMD Raw prediction: {predictions}")
            
            raw_prediction = float(predictions[0][0])
            
            # INVERT THE LOGIC - since model is clearly predicting backwards
            # The model output seems to represent NORMAL confidence, not AMD confidence
            normal_confidence = raw_prediction
            amd_confidence = 1.0 - raw_prediction
            
            # Use a conservative threshold for AMD detection
            threshold = 0.7  # Only classify as AMD if very confident
            
            if amd_confidence >= threshold:
                result = "AMD Detected"
                confidence = amd_confidence
                has_disease = True
            else:
                result = "No AMD"
                confidence = normal_confidence
                has_disease = False
            
            logger.info(f"AMD Prediction: {result} (AMD: {amd_confidence:.4f}, Normal: {normal_confidence:.4f})")
            
            return jsonify({
                'result': result,
                'confidence': confidence,
                'has_disease': has_disease,
                'disease_confidence': amd_confidence,
                'normal_confidence': normal_confidence,
                'success': True
            })
            
        except Exception as e:
            logger.error(f"AMD Prediction error: {e}")
            return jsonify({
                'error': f'Prediction failed: {str(e)}',
                'success': False
            }), 500
        
    except Exception as e:
        logger.error(f"Unexpected error in AMD prediction: {e}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'success': False
        }), 500

@app.route('/api/amd/batch-predict', methods=['POST'])
def batch_predict_amd():
    """
    ISOLATED batch prediction for AMD only
    """
    try:
        if MODELS['amd'] is None:
            return jsonify({
                'error': 'AMD model not loaded',
                'success': False
            }), 503

        files = request.files.getlist('files')
        
        if not files or len(files) == 0:
            return jsonify({
                'error': 'No files uploaded',
                'success': False
            }), 400
        
        results = []
        model = MODELS['amd']
        
        for file in files:
            try:
                if file.filename == '':
                    continue
                
                is_valid, validation_msg = validate_image(file)
                if not is_valid:
                    results.append({
                        'filename': file.filename,
                        'result': 'Invalid File',
                        'confidence': 0,
                        'error': validation_msg,
                        'success': False
                    })
                    continue

                image = Image.open(io.BytesIO(file.read()))
                processed_image = preprocess_for_amd(image)
                
                predictions = model.predict(processed_image, verbose=0)
                amd_confidence = float(predictions[0][0])
                normal_confidence = 1.0 - amd_confidence

                if amd_confidence >= 0.5:
                    result = "AMD Detected"
                    confidence = amd_confidence
                    has_disease = True
                else:
                    result = "No AMD"
                    confidence = normal_confidence
                    has_disease = False
                
                results.append({
                    'filename': file.filename,
                    'result': result,
                    'confidence': confidence,
                    'has_disease': has_disease,
                    'disease_confidence': amd_confidence,
                    'normal_confidence': normal_confidence,
                    'success': True
                })
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {e}")
                results.append({
                    'filename': file.filename,
                    'result': 'Processing Error',
                    'confidence': 0,
                    'error': str(e),
                    'success': False
                })
        
        return jsonify({
            'results': results,
            'success': True,
            'total_processed': len(results),
            'successful': len([r for r in results if r.get('success', False)])
        })
        
    except Exception as e:
        logger.error(f"Unexpected error in AMD batch prediction: {e}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'success': False
        }), 500

# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.route('/')
def home():
    loaded_models = []
    if MODELS['diabetic_retinopathy']:
        loaded_models.append('Diabetic Retinopathy')
    if MODELS['cataract']:
        loaded_models.append('Cataract')
    if MODELS['glaucoma']:
        loaded_models.append('Glaucoma')
    if MODELS['hypertension']:
        loaded_models.append('Hypertension')
    if MODELS['amd']:
        loaded_models.append('AMD')
    
    return jsonify({
        "message": "Retinal Disease Detection API - Isolated Disease Detection",
        "status": "running",
        "loaded_models": loaded_models,
        "endpoints": {
            "diabetic_retinopathy": {
                "single": "/api/diabetic-retinopathy/predict",
                "batch": "/api/diabetic-retinopathy/batch-predict"
            },
            "cataract": {
                "single": "/api/cataract/predict",
                "batch": "/api/cataract/batch-predict"
            },
            "glaucoma": {
                "single": "/api/glaucoma/predict",
                "batch": "/api/glaucoma/batch-predict"
            },
            "hypertension": {
                "single": "/api/hypertension/predict",
                "batch": "/api/hypertension/batch-predict"
            },
            "amd": {
                "single": "/api/amd/predict",
                "batch": "/api/amd/batch-predict"
            }
        }
    })

if __name__ == '__main__':
    os.makedirs('./models', exist_ok=True)
    
    print("\n" + "="*70)
    print("🔍 ISOLATED DISEASE DETECTION SYSTEM")
    print("="*70)
    print("\nChecking for model files...")
    
    dr_path = './models/ensemble_EfficientNetB0.keras'
    cataract_path = './models/cataract_best.keras'
    glaucoma_path = './models/glaucoma_EfficientNetB0.keras'
    hypertension_path = './models/hypertension_best.keras'
    amd_path = './models/amd_classifier_final_improved.keras'
    
    if os.path.exists(dr_path):
        print(f"✅ Diabetic Retinopathy: {dr_path}")
    else:
        print(f"❌ Diabetic Retinopathy: {dr_path} (NOT FOUND)")
    
    if os.path.exists(cataract_path):
        print(f"✅ Cataract: {cataract_path}")
    else:
        print(f"❌ Cataract: {cataract_path} (NOT FOUND)")
    
    if os.path.exists(glaucoma_path):
        print(f"✅ Glaucoma: {glaucoma_path}")
    else:
        print(f"❌ Glaucoma: {glaucoma_path} (NOT FOUND)")
    
    if os.path.exists(hypertension_path):
        print(f"✅ Hypertension: {hypertension_path}")
    else:
        print(f"❌ Hypertension: {hypertension_path} (NOT FOUND)")
    
    if os.path.exists(amd_path):
        print(f"✅ AMD: {amd_path}")
    else:
        print(f"❌ AMD: {amd_path} (NOT FOUND)")
    
    print("="*70 + "\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')