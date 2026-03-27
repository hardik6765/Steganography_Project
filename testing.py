import numpy as np
import tensorflow as tf
from PIL import Image
import os
import sys


MODEL_FILENAME = "steganalysis_model_simple.keras"
IMAGE_SIZE = (128, 128)
THRESHOLD = 0.5

def preprocess_image(filepath):

    try:
        img = Image.open(filepath).convert("RGB").resize(IMAGE_SIZE)
        img_array = np.array(img, dtype=np.float32) / 255.0

        return np.expand_dims(img_array, axis=0)
    except FileNotFoundError:
        print(f"Error: Image not found at path: {filepath}")
        return None
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def run_prediction(image_path):

    if not os.path.exists(MODEL_FILENAME):
        print(f"\n--- ERROR ---")
        print(f"Model '{MODEL_FILENAME}' not found. Please run the training script first.")
        return

    model = tf.keras.models.load_model(MODEL_FILENAME)
    X = preprocess_image(image_path)

    if X is None:
        return

    # Performing the prediction here
    probability = model.predict(X, verbose=0)[0][0]

    # Classify the result
    if probability >= THRESHOLD:
        result = "Stego/Malicious (Hidden Data Detected)"
        confidence = f"{probability*100:.2f}%"
    else:
        result = "Clean Image (No Hidden Data Detected)"
        confidence = f"{(1 - probability)*100:.2f}%"

    print("-" * 40)
    print(f"🔎 Steganalysis Result for: {image_path}")
    print(f"Classification: **{result}**")
    print(f"Confidence: {confidence}")
    print("-" * 40)

if __name__ == "__main__":

    image_path_input = "/content/stego_dataset/stego_clean_001.png" # <<< Change this to your test image path


    run_prediction(image_path_input)
