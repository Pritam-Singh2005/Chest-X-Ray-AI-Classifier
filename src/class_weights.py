from pathlib import Path
import json
from collections import Counter


# ==========================================
# 1. DATASET PATH
# ==========================================

DATASET_DIR = Path(
    r"C:\Users\ps879\Downloads\chest_xray\chest_xray"
)

TRAIN_DIR = DATASET_DIR / "train"


# ==========================================
# 2. CHECK TRAINING DIRECTORY
# ==========================================

print("\n" + "=" * 60)
print("           CLASS IMBALANCE ANALYSIS")
print("=" * 60)

print("\nTraining directory:")
print(TRAIN_DIR)

if not TRAIN_DIR.exists():
    print("\n❌ Training directory not found!")
    exit()

print("✅ Training directory found")


# ==========================================
# 3. COUNT IMAGES
# ==========================================

normal_images = list(
    (TRAIN_DIR / "NORMAL").glob("*")
)

pneumonia_images = list(
    (TRAIN_DIR / "PNEUMONIA").glob("*")
)


normal_count = len([
    f for f in normal_images
    if f.is_file()
])

pneumonia_count = len([
    f for f in pneumonia_images
    if f.is_file()
])


total_images = normal_count + pneumonia_count


# ==========================================
# 4. DISPLAY DATASET DISTRIBUTION
# ==========================================

print("\n" + "=" * 60)
print("             TRAINING DATA")
print("=" * 60)

print("\nNORMAL images:")
print(normal_count)

print("\nPNEUMONIA images:")
print(pneumonia_count)

print("\nTOTAL images:")
print(total_images)


# ==========================================
# 5. CLASS PERCENTAGES
# ==========================================

normal_percentage = (
    normal_count / total_images
) * 100

pneumonia_percentage = (
    pneumonia_count / total_images
) * 100


print("\n" + "=" * 60)
print("             CLASS DISTRIBUTION")
print("=" * 60)

print(
    f"\nNORMAL:    {normal_percentage:.2f}%"
)

print(
    f"PNEUMONIA: {pneumonia_percentage:.2f}%"
)


# ==========================================
# 6. CALCULATE CLASS WEIGHTS
# ==========================================

# Formula:
#
# weight = total_samples /
#          (number_of_classes × class_samples)


number_of_classes = 2


normal_weight = (
    total_images /
    (number_of_classes * normal_count)
)

pneumonia_weight = (
    total_images /
    (number_of_classes * pneumonia_count)
)


# ==========================================
# 7. DISPLAY CLASS WEIGHTS
# ==========================================

print("\n" + "=" * 60)
print("             CLASS WEIGHTS")
print("=" * 60)

print(
    f"\nNORMAL weight:    {normal_weight:.4f}"
)

print(
    f"PNEUMONIA weight: {pneumonia_weight:.4f}"
)


# ==========================================
# 8. CREATE DICTIONARY
# ==========================================

class_weights = {
    0: normal_weight,
    1: pneumonia_weight
}


print("\nClass weights dictionary:")

print(class_weights)


# ==========================================
# 9. SAVE CLASS WEIGHTS
# ==========================================

OUTPUT_PATH = Path(
    r"D:\chest_xray_classifier\models\class_weights.json"
)

with open(
    OUTPUT_PATH,
    "w"
) as file:

    json.dump(
        class_weights,
        file,
        indent=4
    )


# ==========================================
# 10. FINISH
# ==========================================

print("\n" + "=" * 60)
print("       CLASS WEIGHTS SAVED SUCCESSFULLY")
print("=" * 60)

print("\nSaved to:")
print(OUTPUT_PATH)

print("\n✅ Step 8 complete!")