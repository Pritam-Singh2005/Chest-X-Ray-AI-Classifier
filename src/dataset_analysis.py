from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


# ==========================================
# 1. DATASET PATH
# ==========================================

DATASET_DIR = Path(
    r"C:\Users\ps879\Downloads\chest_xray\chest_xray"
)

TRAIN_PATH = DATASET_DIR / "train"
TEST_PATH = DATASET_DIR / "test"
VAL_PATH = DATASET_DIR / "val"


# ==========================================
# 2. CHECK DATASET PATH
# ==========================================

print("\n" + "=" * 55)
print("       CHEST X-RAY DATASET ANALYSIS")
print("=" * 55)

print("\nDataset path:")
print(DATASET_DIR)

print("\nChecking folders...")

print("Dataset exists:   ", DATASET_DIR.exists())
print("Train exists:     ", TRAIN_PATH.exists())
print("Test exists:      ", TEST_PATH.exists())
print("Validation exists:", VAL_PATH.exists())


# Stop if dataset doesn't exist
if not DATASET_DIR.exists():
    print("\n❌ ERROR: Dataset folder not found.")
    print("Please check the dataset path.")
    exit()


# ==========================================
# 3. COUNT IMAGES
# ==========================================

def count_images(folder):

    extensions = {".jpg", ".jpeg", ".png"}

    if not folder.exists():
        return 0

    return sum(
        1
        for file in folder.rglob("*")
        if file.is_file() and file.suffix.lower() in extensions
    )


# ==========================================
# 4. DATASET IMAGE COUNTS
# ==========================================

print("\n" + "=" * 55)
print("              IMAGE COUNTS")
print("=" * 55)

datasets = [
    ("TRAIN", TRAIN_PATH),
    ("VALIDATION", VAL_PATH),
    ("TEST", TEST_PATH)
]

for name, path in datasets:

    normal_count = count_images(path / "NORMAL")
    pneumonia_count = count_images(path / "PNEUMONIA")

    total = normal_count + pneumonia_count

    print(f"\n{name}")
    print("-" * 30)
    print(f"NORMAL:     {normal_count}")
    print(f"PNEUMONIA:  {pneumonia_count}")
    print(f"TOTAL:      {total}")


# ==========================================
# 5. GET IMAGE FILES
# ==========================================

image_extensions = {".jpg", ".jpeg", ".png"}

normal_images = [
    file
    for file in (TRAIN_PATH / "NORMAL").iterdir()
    if file.is_file() and file.suffix.lower() in image_extensions
]

pneumonia_images = [
    file
    for file in (TRAIN_PATH / "PNEUMONIA").iterdir()
    if file.is_file() and file.suffix.lower() in image_extensions
]


print("\n" + "=" * 55)
print("             SAMPLE IMAGES")
print("=" * 55)

print("\nNORMAL images found:", len(normal_images))
print("PNEUMONIA images found:", len(pneumonia_images))


# ==========================================
# 6. DISPLAY SAMPLE X-RAYS
# ==========================================

if normal_images and pneumonia_images:

    normal_image = Image.open(normal_images[0])
    pneumonia_image = Image.open(pneumonia_images[0])

    plt.figure(figsize=(10, 5))

    # NORMAL
    plt.subplot(1, 2, 1)

    plt.imshow(normal_image, cmap="gray")

    plt.title("NORMAL")
    plt.axis("off")

    # PNEUMONIA
    plt.subplot(1, 2, 2)

    plt.imshow(pneumonia_image, cmap="gray")

    plt.title("PNEUMONIA")
    plt.axis("off")

    plt.tight_layout()

    plt.show()

else:

    print("\n⚠️ Could not find sample images.")
    print("Please check the NORMAL and PNEUMONIA folders.")


# ==========================================
# 7. IMAGE INFORMATION
# ==========================================

print("\n" + "=" * 55)
print("            IMAGE INFORMATION")
print("=" * 55)


if normal_images:

    image = Image.open(normal_images[0])

    print("\nNORMAL image")
    print("File:", normal_images[0].name)
    print("Size:", image.size)
    print("Mode:", image.mode)


if pneumonia_images:

    image = Image.open(pneumonia_images[0])

    print("\nPNEUMONIA image")
    print("File:", pneumonia_images[0].name)
    print("Size:", image.size)
    print("Mode:", image.mode)


# ==========================================
# 8. CLASS DISTRIBUTION
# ==========================================

normal_count = count_images(TRAIN_PATH / "NORMAL")
pneumonia_count = count_images(TRAIN_PATH / "PNEUMONIA")

print("\n" + "=" * 55)
print("          CLASS DISTRIBUTION")
print("=" * 55)

print(f"\nNORMAL images:    {normal_count}")
print(f"PNEUMONIA images: {pneumonia_count}")


# ==========================================
# 9. CLASS DISTRIBUTION GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.bar(
    ["NORMAL", "PNEUMONIA"],
    [normal_count, pneumonia_count]
)

plt.title("Training Dataset Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")

plt.tight_layout()

plt.show()


# ==========================================
# 10. IMBALANCE CHECK
# ==========================================

if normal_count > 0 and pneumonia_count > 0:

    ratio = pneumonia_count / normal_count

    print("\nPneumonia / Normal ratio:", round(ratio, 2))

    if ratio > 1.5:

        print("\n⚠️ Dataset is imbalanced.")
        print("There are significantly more PNEUMONIA images.")

    else:

        print("\nDataset is relatively balanced.")


# ==========================================
# 11. ANALYSIS COMPLETE
# ==========================================

print("\n" + "=" * 55)
print("           ANALYSIS COMPLETE")
print("=" * 55)

print("\nNext step:")
print("Prepare the dataset for model training.")
print()