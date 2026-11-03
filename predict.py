"""
Load the trained model and predict Normal / Pneumonia for a single
chest X-ray image, with a confidence score.
"""

import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (224, 224)
MODEL_PATH = "model/medical_model.h5"
LABELS_PATH = "model/labels.txt"


def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def preprocess_image(image_path: str):
    img = load_img(image_path, target_size=IMG_SIZE)
    arr = img_to_array(img)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def predict(image_path: str):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Train it first with `python train.py`.")

    labels = load_labels()
    model = load_model(MODEL_PATH)

    x = preprocess_image(image_path)
    prob = float(model.predict(x, verbose=0)[0][0])  # probability of class index 1

    if prob >= 0.5:
        prediction = labels[1]
        confidence = prob
    else:
        prediction = labels[0]
        confidence = 1 - prob

    return {"prediction": prediction, "confidence": confidence}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to chest X-ray image")
    args = parser.parse_args()

    result = predict(args.image)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
