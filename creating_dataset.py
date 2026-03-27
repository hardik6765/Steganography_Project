import os
import numpy as np
from PIL import Image

COVER_IMAGE_DIR = "cover_images"
STEGO_IMAGE_DIR = "stego_dataset"
BITS_TO_EMBED = 1
PAYLOAD_SIZE = 1024

def load_image(filepath):
    img = Image.open(filepath).convert("RGB")
    return np.array(img)

def save_image(array, filepath):
    img = Image.fromarray(array.astype('uint8'), 'RGB')
    img.save(filepath)

def lsb_embed(cover_array, payload_size, bits=1):
    stego_array = cover_array.copy()
    required_bits = payload_size * 8
    flat_array = stego_array.flatten()

    if len(flat_array) < required_bits:

        return None

    random_payload = np.random.randint(0, 2, size=required_bits, dtype=np.uint8)


    for i in range(required_bits):

        pixel_val = flat_array[i]

        mask = (0xFF << bits) & 0xFF
        new_val = (pixel_val & mask) | random_payload[i]

        flat_array[i] = new_val


    return flat_array.reshape(stego_array.shape)

def generate_dataset():
    os.makedirs(COVER_IMAGE_DIR, exist_ok=True)
    os.makedirs(STEGO_IMAGE_DIR, exist_ok=True)

    if not os.listdir(COVER_IMAGE_DIR):
        print(f"--- ERROR ---")
        print(f"Please place your initial clean images (e.g., .png, .jpg) in the '{COVER_IMAGE_DIR}' directory.")
        print(f"Example: Download 100 images and put them there.")
        return

    print(f"Starting dataset generation from '{COVER_IMAGE_DIR}'...")

    count = 0
    for filename in os.listdir(COVER_IMAGE_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            cover_path = os.path.join(COVER_IMAGE_DIR, filename)
            stego_path = os.path.join(STEGO_IMAGE_DIR, f"stego_{filename}")

            try:
                cover_array = load_image(cover_path)
                stego_array = lsb_embed(cover_array, PAYLOAD_SIZE, BITS_TO_EMBED)

                if stego_array is not None:

                    save_image(stego_array, stego_path)

                    count += 1
            except Exception as e:
                print(f"Could not process {filename}: {e}")

    print("-" * 30)
    print(f"Finished! Generated {count} stego images.")
    print(f"Your dataset is ready in two locations:")
    print(f"  - Clean Images (Label 0): {COVER_IMAGE_DIR}/")
    print(f"  - Stego Images (Label 1): {STEGO_IMAGE_DIR}/")

if __name__ == "__main__":
    generate_dataset()
