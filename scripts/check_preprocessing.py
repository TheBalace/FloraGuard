# scripts/check_preprocessing.py
#
# Visual sanity check for BOTH My_Garden and PlantVillage processed images.
# Shows raw vs processed side by side for a random sample from each class.
#
# What to look for:
#   ✓ Leaf is the main subject, tightly cropped
#   ✓ Background is reduced
#   ✓ Disease spots are still clearly visible
#   ✗ Pure black image → masking failed completely
#   ✗ Crop is of background, not leaf → contour detected wrong blob

import cv2
import matplotlib.pyplot as plt
import os
import random

# --- Path setup ---
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR       = os.path.join(PROJECT_ROOT, "raw_data")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed_data")

# How many images to sample per class
SAMPLE_SIZE = 3

def check_category(raw_folder, processed_folder, title):
    """
    For a given class folder, picks SAMPLE_SIZE random images,
    shows raw vs processed side by side, and saves to reports/.
    
    Parameters:
        raw_folder       (str): Path to raw images for this class
        processed_folder (str): Path to processed images for this class
        title            (str): Label for the plot title
    """

    # Get all image filenames in the raw folder
    all_files = sorted([
        f for f in os.listdir(raw_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if len(all_files) == 0:
        print(f"  ✗ No images found in {raw_folder}")
        return

    # random.sample picks SAMPLE_SIZE unique items from the list
    # min() ensures we don't request more samples than files available
    sample_files = random.sample(all_files, min(SAMPLE_SIZE, len(all_files)))

    # Create figure: 2 rows × SAMPLE_SIZE columns
    fig, axes = plt.subplots(2, len(sample_files), figsize=(5 * len(sample_files), 6))
    fig.suptitle(f"{title}\nTop: Raw    Bottom: Processed", fontsize=12, fontweight='bold')

    # If only 1 sample, axes won't be 2D — this fixes that edge case
    if len(sample_files) == 1:
        axes = [[axes[0]], [axes[1]]]

    for i, filename in enumerate(sample_files):
        raw_path       = os.path.join(raw_folder,       filename)
        processed_path = os.path.join(processed_folder, filename)

        # Check processed file exists (it should, but just in case)
        if not os.path.exists(processed_path):
            print(f"  ✗ Missing processed file: {filename}")
            continue

        # Load both — convert BGR→RGB for matplotlib display
        raw_img       = cv2.cvtColor(cv2.imread(raw_path),       cv2.COLOR_BGR2RGB)
        processed_img = cv2.cvtColor(cv2.imread(processed_path), cv2.COLOR_BGR2RGB)

        # Top row: raw
        axes[0][i].imshow(raw_img)
        axes[0][i].set_title(filename, fontsize=7)
        axes[0][i].axis('off')

        # Bottom row: processed
        axes[1][i].imshow(processed_img)
        axes[1][i].set_title("Processed (224×224)", fontsize=7)
        axes[1][i].axis('off')

    plt.tight_layout()

    # Save to reports/ — replace slashes in title with underscores for filename safety
    safe_title  = title.replace("/", "_").replace(" ", "_")
    report_path = os.path.join(PROJECT_ROOT, "reports", f"check_{safe_title}.png")
    plt.savefig(report_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: reports/check_{safe_title}.png")
    plt.show()
    plt.close()  # Free memory — important when looping many classes


# ===========================================================
# PART 1: Check My_Garden classes
# ===========================================================

print("\n" + "=" * 50)
print("  Checking My_Garden classes")
print("=" * 50)

garden_classes = ["Rose_Disease", "Hibiscus_Healthy", "Parijat_Healthy"]

for class_name in garden_classes:
    raw_path       = os.path.join(RAW_DIR,       "My_Garden", class_name)
    processed_path = os.path.join(PROCESSED_DIR, "My_Garden", class_name)
    print(f"\n→ {class_name}")
    check_category(raw_path, processed_path, f"My_Garden / {class_name}")


# ===========================================================
# PART 2: Check PlantVillage classes — random sample per class
# ===========================================================

print("\n" + "=" * 50)
print("  Checking PlantVillage classes")
print("=" * 50)

plantvillage_raw       = os.path.join(RAW_DIR,       "PlantVillage")
plantvillage_processed = os.path.join(PROCESSED_DIR, "PlantVillage")

# Discover all class folders automatically
pv_classes = sorted([
    item for item in os.listdir(plantvillage_raw)
    if os.path.isdir(os.path.join(plantvillage_raw, item))
])

for class_name in pv_classes:
    raw_path       = os.path.join(plantvillage_raw,       class_name)
    processed_path = os.path.join(plantvillage_processed, class_name)
    print(f"\n→ {class_name}")
    check_category(raw_path, processed_path, f"PlantVillage / {class_name}")


print("\n✅ Visual check complete.")
print("   All comparison images saved to reports/ folder.")
print("   Review them and confirm leaves are correctly cropped in processed versions.")