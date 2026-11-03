"""
Train a transfer-learning CNN (MobileNetV2) to classify chest X-rays
as Normal or Pneumonia.

Expects dataset organized as:
  dataset/train/NORMAL/*.jpeg
  dataset/train/PNEUMONIA/*.jpeg
  dataset/val/NORMAL/*.jpeg
  dataset/val/PNEUMONIA/*.jpeg

No JSON is used - class labels are saved as plain text (labels.txt),
model saved as .h5.
"""

import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "medical_model.h5")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.txt")

os.makedirs(MODEL_DIR, exist_ok=True)


def build_model():
    base = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False  # freeze pretrained ImageNet layers

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)  # binary: Normal vs Pneumonia

    model = Model(inputs=base.input, outputs=output)
    return model


def main():
    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True,
    )
    val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_data = train_gen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary"
    )
    val_data = val_gen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary"
    )

    # class_indices e.g. {'NORMAL': 0, 'PNEUMONIA': 1} -> save order as plain text
    labels_sorted = sorted(train_data.class_indices, key=train_data.class_indices.get)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        for label in labels_sorted:
            f.write(label + "\n")

    model = build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Classes (label order): {labels_sorted}")


if __name__ == "__main__":
    main()
