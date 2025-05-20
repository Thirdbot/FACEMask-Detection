import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pickle

# Initialize paths and parameters
DATASET_PATH = "dataset/data"
INIT_LR = 1e-4
EPOCHS = 15
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

# Load and preprocess dataset
print("[INFO] Loading images...")
categories = ["with_mask", "without_mask"]
data, labels = [], []

for category in categories:
    path = os.path.join(DATASET_PATH, category)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        image = plt.imread(img_path)
        image = np.resize(image, (*IMG_SIZE, 3)) / 255.0
        data.append(image)
        labels.append(category)

# Convert to NumPy arrays and encode labels
data = np.array(data, dtype="float32")
labels = np.array(labels)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = np.hstack((labels, 1 - labels))

# Split data
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

# Data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2,
                         height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

# Build model
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(*IMG_SIZE, 3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten()(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze base model layers
for layer in baseModel.layers:
    layer.trainable = False

# Compile model
print("[INFO] Compiling model...")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train model
print("[INFO] Training model...")
H = model.fit(aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
              validation_data=(testX, testY), epochs=EPOCHS, verbose=1)

# Evaluate model
print("[INFO] Evaluating model...")
predIdxs = model.predict(testX, batch_size=BATCH_SIZE)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

# Save model
print("[INFO] Saving model...")
model.save("mask_detector.h5") 

# Save Label Binarizer
print("[INFO] Saving label binarizer...")
with open("label_binarizer.pickle", "wb") as f:
    pickle.dump(lb, f)
