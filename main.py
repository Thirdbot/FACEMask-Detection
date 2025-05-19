import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks, applications

# Step 2: Data Preparation
DATASET_DIR = 'cleaned_dataset'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,          # Increased rotation
    width_shift_range=0.2,      # Increased shift
    height_shift_range=0.2,     # Increased shift
    shear_range=0.15,          # Increased shear
    zoom_range=0.2,            # Increased zoom
    horizontal_flip=True,
    brightness_range=[0.8,1.2], # Added brightness variation
    fill_mode='constant',       # Changed to constant
    cval=0                      # Black background for augmentation
)

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Step 3: Model Selection & Training (Transfer Learning)
base_model = applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
# Unfreeze some layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-4]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

# Calculate class weights
total_samples = sum([len(files) for r, d, files in os.walk(DATASET_DIR)])
n_samples_per_class = [len(files) for r, d, files in os.walk(DATASET_DIR) if files]
class_weights = {i: total_samples / (len(n_samples_per_class) * count) 
                for i, count in enumerate(n_samples_per_class)}

# Use a lower learning rate for fine-tuning
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Add more callbacks for better training
callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,          # Reduced from 5
        restore_best_weights=True
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,          # Reduced from 3
        min_lr=0.00001
    ),
    callbacks.ModelCheckpoint(
        'facemask_model.h5',
        save_best_only=True,
        monitor='val_loss'
    )
]

history = model.fit(
    train_gen,
    epochs=15,              # Reduced from 30
    validation_data=val_gen,
    callbacks=callbacks_list,
    class_weight=class_weights
)

# Step 4: Model Evaluation
evaluation = model.evaluate(val_gen)
print("\nEvaluation Results:")
for metric_name, value in zip(model.metrics_names, evaluation):
    print(f"{metric_name}: {value:.4f}")

# Optionally save the final model
model.save('facemask_model.h5')
