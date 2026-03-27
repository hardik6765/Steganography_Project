import os
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

COVER_DIR = "cover_images"
STEGO_DIR = "stego_dataset"
MODEL_FILENAME = "steganalysis_model_simple.keras"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 8

def load_data():

    data, labels = [], []


    datasets = [(COVER_DIR, 0), (STEGO_DIR, 1)]

    print("Loading images...")
    for directory, label in datasets:
        for filename in os.listdir(directory):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    # convert to normalized array
                    img_path = os.path.join(directory, filename)
                    img = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
                    img_array = np.array(img, dtype=np.float32) / 255.0

                    data.append(img_array)
                    labels.append(label)
                except Exception as e:
                    print(f"Skipping {filename}: {e}")
                    continue

    if not data:
        print("ERROR: No images found. Check directory paths.")
        return None, None, None, None


    X = np.array(data)
    y = np.array(labels)


    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def build_simple_model(input_shape):
    model = Sequential([

        Conv2D(16, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling2D((2, 2)),

        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_save():
    X_train, X_test, y_train, y_test = load_data()

    if X_train is None:
        return


    input_shape = X_train.shape[1:]


    model = build_simple_model(input_shape)
    print("\n--- Model Summary ---")
    model.summary()


    print("\n--- Starting Training ---")
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )


    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("\n" + "=" * 30)
    print(f" Training Finished. Test Accuracy: {accuracy*100:.2f}%")
    print("=" * 30)

    model.save(MODEL_FILENAME)
    print(f"Model saved as '{MODEL_FILENAME}'")

if __name__ == "__main__":
    train_and_save()
