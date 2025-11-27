import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import os

# Configuration
IMG_SIZE = 380  # EfficientNetB4 optimal size
BATCH_SIZE = 16
EPOCHS = 150
INITIAL_LR = 0.0001

# Data paths
train_dir = 'amd/train'
val_dir = 'amd/val'

print("="*80)
print("AMD (AGE-RELATED MACULAR DEGENERATION) CLASSIFICATION MODEL")
print("="*80)

# Advanced Data Augmentation for Medical Images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.2,
    zoom_range=0.25,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='reflect'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"\nDataset Information:")
print(f"  Class mapping: {train_generator.class_indices}")
print(f"  Training samples: {train_generator.samples}")
print(f"  Validation samples: {val_generator.samples}")

# Analyze class distribution
train_class_counts = np.bincount(train_generator.classes)
val_class_counts = np.bincount(val_generator.classes)

print(f"\nClass Distribution:")
print(f"  Training - AMD: {train_class_counts[0]}, No-AMD: {train_class_counts[1]}")
print(f"  Validation - AMD: {val_class_counts[0]}, No-AMD: {val_class_counts[1]}")

# Calculate class weights for imbalanced data
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = dict(enumerate(class_weights))
print(f"\nClass Weights: {class_weight_dict}")

# Build Transfer Learning Model with EfficientNetB4
def create_amd_model():
    """
    Creates an AMD classification model using EfficientNetB4
    Optimized for medical image classification
    """
    # Load pre-trained EfficientNetB4
    base_model = EfficientNetB4(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling='avg'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build custom top layers
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Base model
    x = base_model(inputs, training=False)
    
    # Custom classification head
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256, activation='relu', 
                     kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid', name='amd_prediction')(x)
    
    model = keras.Model(inputs, outputs, name='AMD_Classifier')
    
    return model, base_model

# Create model
model, base_model = create_amd_model()

# Focal Loss for better handling of class imbalance
class FocalLoss(keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, name='focal_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Binary cross entropy
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Focal loss modulation
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1 - p_t, self.gamma)
        
        # Alpha weighting
        alpha_weight = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        
        return alpha_weight * focal_weight * bce

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR),
    loss=FocalLoss(alpha=0.25, gamma=2.0),
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
        keras.metrics.AUC(name='prc', curve='PR')  # Precision-Recall AUC
    ]
)

print("\n" + "="*80)
print("MODEL ARCHITECTURE")
print("="*80)
model.summary()

# Calculate training steps
steps_per_epoch = max(train_generator.samples // BATCH_SIZE, 10)
validation_steps = max(val_generator.samples // BATCH_SIZE, 5)

print(f"\nTraining Configuration:")
print(f"  Steps per epoch: {steps_per_epoch}")
print(f"  Validation steps: {validation_steps}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Initial learning rate: {INITIAL_LR}")

# Callbacks for Phase 1
callbacks_phase1 = [
    EarlyStopping(
        monitor='val_auc',
        patience=20,
        mode='max',
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-8,
        verbose=1
    ),
    ModelCheckpoint(
        'best_amd_model_phase1.keras',
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    )
]

# PHASE 1: Train with frozen base
print("\n" + "="*80)
print("PHASE 1: Training with Frozen Base Model")
print("="*80)

history1 = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    validation_data=val_generator,
    validation_steps=validation_steps,
    callbacks=callbacks_phase1,
    class_weight=class_weight_dict,
    verbose=1
)

# PHASE 2: Fine-tune the model
print("\n" + "="*80)
print("PHASE 2: Fine-Tuning (Unfreezing Top Layers)")
print("="*80)

# Unfreeze top layers
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

print(f"Trainable layers: {sum([1 for layer in model.layers if layer.trainable])}")

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR/10),
    loss=FocalLoss(alpha=0.25, gamma=2.0),
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
        keras.metrics.AUC(name='prc', curve='PR')
    ]
)

callbacks_phase2 = [
    EarlyStopping(
        monitor='val_auc',
        patience=15,
        mode='max',
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-9,
        verbose=1
    ),
    ModelCheckpoint(
        'best_amd_model_final.keras',
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    )
]

history2 = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=100,
    validation_data=val_generator,
    validation_steps=validation_steps,
    callbacks=callbacks_phase2,
    class_weight=class_weight_dict,
    verbose=1
)

# Combine histories
history_combined = {
    'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
    'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
    'loss': history1.history['loss'] + history2.history['loss'],
    'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
    'auc': history1.history['auc'] + history2.history['auc'],
    'val_auc': history1.history['val_auc'] + history2.history['val_auc']
}

# Evaluate on validation set
print("\n" + "="*80)
print("FINAL EVALUATION ON VALIDATION SET")
print("="*80)

val_loss, val_acc, val_prec, val_rec, val_auc_score, val_prc = model.evaluate(
    val_generator, 
    verbose=1
)

print(f"\nValidation Metrics:")
print(f"  Accuracy:  {val_acc*100:.2f}%")
print(f"  Precision: {val_prec*100:.2f}%")
print(f"  Recall:    {val_rec*100:.2f}%")
print(f"  F1-Score:  {2*(val_prec*val_rec)/(val_prec+val_rec+1e-7)*100:.2f}%")
print(f"  AUC-ROC:   {val_auc_score*100:.2f}%")
print(f"  AUC-PR:    {val_prc*100:.2f}%")

# Generate predictions for detailed analysis
print("\nGenerating predictions for detailed analysis...")
val_generator.reset()
predictions = model.predict(val_generator, verbose=1)
predicted_classes = (predictions > 0.5).astype(int).flatten()
true_classes = val_generator.classes

# Classification Report
print("\n" + "="*80)
print("DETAILED CLASSIFICATION REPORT")
print("="*80)
print(classification_report(
    true_classes, 
    predicted_classes,
    target_names=['AMD', 'No-AMD'],
    digits=4
))

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
print("\nConfusion Matrix:")
print(cm)
print(f"\nBreakdown:")
print(f"  True AMD (correctly identified):     {cm[0,0]}")
print(f"  AMD misclassified as No-AMD:         {cm[0,1]}")
print(f"  No-AMD misclassified as AMD:         {cm[1,0]}")
print(f"  True No-AMD (correctly identified):  {cm[1,1]}")

# Plot Training History
plt.figure(figsize=(18, 5))

# Accuracy
plt.subplot(1, 3, 1)
plt.plot(history_combined['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(history_combined['val_accuracy'], label='Val Accuracy', linewidth=2)
plt.axvline(x=len(history1.history['accuracy']), color='r', linestyle='--', 
            label='Fine-tuning starts', alpha=0.7)
plt.title('Model Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Loss
plt.subplot(1, 3, 2)
plt.plot(history_combined['loss'], label='Train Loss', linewidth=2)
plt.plot(history_combined['val_loss'], label='Val Loss', linewidth=2)
plt.axvline(x=len(history1.history['loss']), color='r', linestyle='--',
            label='Fine-tuning starts', alpha=0.7)
plt.title('Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# AUC
plt.subplot(1, 3, 3)
plt.plot(history_combined['auc'], label='Train AUC', linewidth=2)
plt.plot(history_combined['val_auc'], label='Val AUC', linewidth=2)
plt.axvline(x=len(history1.history['auc']), color='r', linestyle='--',
            label='Fine-tuning starts', alpha=0.7)
plt.title('Model AUC-ROC', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('amd_training_history.png', dpi=300, bbox_inches='tight')
print("\n✓ Training history saved as 'amd_training_history.png'")
plt.show()

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['AMD', 'No-AMD'],
            yticklabels=['AMD', 'No-AMD'],
            annot_kws={'size': 16})
plt.title('Confusion Matrix - AMD Classification', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.savefig('amd_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrix saved as 'amd_confusion_matrix.png'")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(true_classes, predictions)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - AMD Classification', fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('amd_roc_curve.png', dpi=300, bbox_inches='tight')
print("✓ ROC curve saved as 'amd_roc_curve.png'")
plt.show()

# Save final model
model.save('amd_classifier_final.keras')
print("\n" + "="*80)
print("✓ Final model saved as 'amd_classifier_final.keras'")
print("✓ Best model saved as 'best_amd_model_final.keras'")
print("="*80)

# Summary
print("\n" + "="*80)
print("TRAINING COMPLETE - AMD CLASSIFICATION MODEL")
print("="*80)
print(f"\nFinal Performance:")
print(f"  • Validation Accuracy: {val_acc*100:.2f}%")
print(f"  • AUC-ROC: {val_auc_score*100:.2f}%")
print(f"  • Sensitivity (Recall): {val_rec*100:.2f}%")
print(f"  • Specificity: {cm[1,1]/(cm[1,1]+cm[1,0])*100:.2f}%")
print(f"\nFiles Generated:")
print(f"  • amd_classifier_final.keras")
print(f"  • best_amd_model_final.keras")
print(f"  • amd_training_history.png")
print(f"  • amd_confusion_matrix.png")
print(f"  • amd_roc_curve.png")
print("="*80)