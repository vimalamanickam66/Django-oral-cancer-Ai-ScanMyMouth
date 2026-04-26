import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

print("\n" + "=" * 60)
print("   ScanMyMouth AI - Model Training")
print("=" * 60)

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_clean")
TRAIN_DIR   = os.path.join(DATASET_DIR, "train")
VAL_DIR     = os.path.join(DATASET_DIR, "val")
TEST_DIR    = os.path.join(DATASET_DIR, "test")

MODEL_DIR  = os.path.join(BASE_DIR, "oral_detection", "models_dir")
os.makedirs(MODEL_DIR, exist_ok=True)

# FIXED: Save as .keras to avoid BatchNormalization serialization issues
MODEL_PATH = os.path.join(MODEL_DIR, "oral_cancer_model.keras")

print(f"\nDataset  : {DATASET_DIR}")
print(f"Model    : {MODEL_PATH}")

# --------------------------------------------------
# Check folders exist
# --------------------------------------------------
for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if not os.path.exists(folder):
        print(f"\nERROR: Folder not found -> {folder}")
        print("Run rebuild_dataset_to_jpg.py first.")
        exit()

# --------------------------------------------------
# Settings
# --------------------------------------------------
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS     = 50

# --------------------------------------------------
# Data generators
# FIXED: class_mode='binary' to match sigmoid output
# FIXED: No rescale — MobileNetV2 preprocess_input is inside the model
# --------------------------------------------------
print("\nLoading datasets...")

train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.20,
    shear_range=0.10,
    brightness_range=[0.80, 1.20],
    channel_shift_range=20.0,
    fill_mode='nearest',
)
val_datagen = ImageDataGenerator()

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    classes=['cancerous', 'normal'],
)
val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False,
    classes=['cancerous', 'normal'],
)

print(f"\nClasses : {train_gen.class_indices}")
print(f"Train   : {train_gen.samples} images")
print(f"Val     : {val_gen.samples} images")

# --------------------------------------------------
# FIXED: Class weights to handle imbalanced dataset
# --------------------------------------------------
cw           = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_gen.classes)
class_weight = dict(enumerate(cw))
print(f"\nClass weights: {class_weight}")

# --------------------------------------------------
# Build model — architecture matches ai_engine.py exactly
# --------------------------------------------------
print("\nBuilding MobileNetV2 model...")

base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base_model.trainable = False

inputs  = tf.keras.Input(shape=(224, 224, 3), name='oral_image_input')
x       = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x       = base_model(x, training=False)
x       = layers.GlobalAveragePooling2D(name='gap')(x)
x       = layers.Dropout(0.3, name='dropout')(x)
outputs = layers.Dense(1, activation="sigmoid", name='predictions')(x)

model = models.Model(inputs, outputs, name='ScanMyMouth_MobileNetV2')

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ],
)
model.summary()

# --------------------------------------------------
# Phase 1 callbacks
# FIXED: Monitor val_auc instead of val_loss
# --------------------------------------------------
callbacks_p1 = [
    EarlyStopping(monitor="val_auc", patience=12, mode='max',
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor="val_auc", save_best_only=True,
                    mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                      min_lr=1e-7, verbose=1),
    tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1),
    tf.keras.callbacks.CSVLogger('training_log.csv'),
]

# --------------------------------------------------
# Phase 1: Train classification head
# --------------------------------------------------
print("\nPhase 1: Training classification head (backbone frozen)...\n")

model.fit(
    train_gen,
    steps_per_epoch  = max(1, train_gen.samples // BATCH_SIZE),
    validation_data  = val_gen,
    validation_steps = max(1, val_gen.samples  // BATCH_SIZE),
    epochs           = EPOCHS,
    callbacks        = callbacks_p1,
    class_weight     = class_weight,
)

# --------------------------------------------------
# Phase 2: Fine-tune last 30 layers
# --------------------------------------------------
print("\nPhase 2: Fine-tuning last 30 layers of MobileNetV2...\n")

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ],
)

train_gen.reset()
val_gen.reset()

callbacks_p2 = [
    EarlyStopping(monitor="val_auc", patience=8, mode='max',
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor="val_auc", save_best_only=True,
                    mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                      min_lr=1e-8, verbose=1),
]

model.fit(
    train_gen,
    steps_per_epoch  = max(1, train_gen.samples // BATCH_SIZE),
    validation_data  = val_gen,
    validation_steps = max(1, val_gen.samples  // BATCH_SIZE),
    epochs           = 20,
    callbacks        = callbacks_p2,
    class_weight     = class_weight,
)

# --------------------------------------------------
# Evaluate on test set
# --------------------------------------------------
print("\nEvaluating on test dataset...\n")

test_datagen = ImageDataGenerator()
test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False,
    classes=['cancerous', 'normal'],
)

results = model.evaluate(test_gen, steps=max(1, test_gen.samples // BATCH_SIZE))
print(f"\nTest Loss      : {results[0]:.4f}")
print(f"Test Accuracy  : {results[1]:.4f}")
print(f"Test AUC       : {results[2]:.4f}")
print(f"Test Precision : {results[3]:.4f}")
print(f"Test Recall    : {results[4]:.4f}")

# --------------------------------------------------
# Save final model
# --------------------------------------------------
model.save(MODEL_PATH)

print("\n" + "=" * 60)
print(" Training completed!")
print(f" Model saved: {MODEL_PATH}")
print("=" * 60)
print("\nUpdate settings.py:")
print("  AI_MODEL_PATH = BASE_DIR / 'oral_detection' / 'models_dir' / 'oral_cancer_model.keras'")