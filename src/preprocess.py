import os
import glob
import h5py
import numpy as np
from PIL import Image


# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_dataset")

# Dataset label mapping
LABEL_MAP = {
    1: "Meningioma",
    2: "Glioma",
    3: "Pituitary",
}


def normalize_image(image):
    image = image.astype(np.float32)

    min_val = image.min()
    max_val = image.max()

    if max_val > min_val:
        image = (image - min_val) / (max_val - min_val)
    else:
        image = np.zeros_like(image)

    image = (image * 255).astype(np.uint8)

    return image


def process_file(mat_path):
    with h5py.File(mat_path, "r") as f:
        cjdata = f["cjdata"]

        image = np.array(cjdata["image"])
        label = int(cjdata["label"][()][0][0])

    if label not in LABEL_MAP:
        print(f"Skipping unknown label: {label} -> {mat_path}")
        return False

    class_name = LABEL_MAP[label]

    output_class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    image = normalize_image(image)

    filename = os.path.splitext(os.path.basename(mat_path))[0] + ".jpg"
    output_path = os.path.join(output_class_dir, filename)

    Image.fromarray(image).save(
        output_path,
        format="JPEG",
        quality=95
    )

    return True


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mat_files = []

    for set_name in ["bt_set1", "bt_set2", "bt_set3", "bt_set4"]:
        set_dir = os.path.join(DATASET_DIR, set_name)

        files = [os.path.join(set_dir, filename) for filename in os.listdir(set_dir) if filename.lower().endswith(".mat")]
        mat_files.extend(files)

    mat_files.sort()

    print(f"Found {len(mat_files)} .mat files")

    processed = 0

    for i, mat_path in enumerate(mat_files, start=1):
        try:
            if process_file(mat_path):
                processed += 1

            if i % 100 == 0:
                print(f"Processed: {i}/{len(mat_files)}")

        except Exception as e:
            print(f"ERROR: {mat_path}")
            print(e)

    print("\nPreprocessing complete!")
    print(f"Successfully processed: {processed}/{len(mat_files)}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
