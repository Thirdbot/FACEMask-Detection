import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# --- Configuration ---
DATA_DIR = "dataset/cleaned"  # contains subfolders 'with_mask', 'without_mask'
MODEL_PATH = "mask_detector.keras"  # Use the newer .keras format
PLOT_PATH = "training_plot.png"
INIT_LR = 1e-4
EPOCHS = 13
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
VALIDATION_SPLIT = 0.2

# --- Data Generators ---
train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=VALIDATION_SPLIT
)

train_gen = train_aug.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_gen = train_aug.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# --- Build Model ---
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(*IMG_SIZE, 3)))
head = baseModel.output
head = AveragePooling2D(pool_size=(7, 7))(head)
head = Flatten()(head)
head = Dense(128, activation="relu")(head)
head = Dropout(0.5)(head)
head = Dense(1, activation="sigmoid")(head)

model = Model(inputs=baseModel.input, outputs=head)

# Freeze base layers
for layer in baseModel.layers:
    layer.trainable = False

# Compile
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True),  # Removed save_format
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)
]

# Adjust class weights
class_weights = {0: 1.0, 1: 2.0}  # Adjust weights based on class distribution

# Fine-tune the model
for layer in baseModel.layers[-20:]:  # Unfreeze the last 20 layers
    layer.trainable = True
opt = Adam(learning_rate=INIT_LR / 10)  # Lower learning rate
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# --- Train ---
H = model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // BATCH_SIZE,
    validation_data=val_gen,
    validation_steps=(val_gen.samples + BATCH_SIZE - 1) // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,  # Add class weights
    verbose=1
)

# Save the final model explicitly in .keras format
model.save(MODEL_PATH)

# --- Evaluate ---
print("[INFO] Evaluating model...")
# Predict on validation set
val_gen.reset()
predIdx = (model.predict(val_gen, steps=(val_gen.samples + BATCH_SIZE - 1) // BATCH_SIZE) > 0.5).astype("int32").ravel()
trueIdx = val_gen.classes[:len(predIdx)]

print(classification_report(trueIdx, predIdx,
      target_names=list(val_gen.class_indices.keys())))
print("Confusion Matrix:\n", confusion_matrix(trueIdx, predIdx))

# --- Plot Training Curves ---
plt.style.use("ggplot")
plt.figure()
plt.plot(H.history['loss'], label='train_loss')
plt.plot(H.history['val_loss'], label='val_loss')
plt.plot(H.history['accuracy'], label='train_acc')
plt.plot(H.history['val_accuracy'], label='val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='lower left')
plt.savefig(PLOT_PATH)
plt.close()

print(f"[INFO] Model saved to {MODEL_PATH}")
print(f"[INFO] Training plot saved to {PLOT_PATH}")
