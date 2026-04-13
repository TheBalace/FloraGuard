import cv2
import matplotlib.pyplot as plt
import os
import random

PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR       = os.path.join(PROJECT_ROOT, "raw_data")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed_data")

SAMPLE_SIZE = 3

def check_category(raw_folder, processed_folder, title):

    all_files = sorted([
        f for f in os.listdir(raw_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if len(all_files) == 0:
        print(f"  ✗ No images found in {raw_folder}")
        return


    sample_files = random.sample(all_files, min(SAMPLE_SIZE, len(all_files)))

    fig, axes = plt.subplots(2, len(sample_files), figsize=(5 * len(sample_files), 6))
    fig.suptitle(f"{title}\nTop: Raw    Bottom: Processed", fontsize=12, fontweight='bold')

    if len(sample_files) == 1:
        axes = [[axes[0]], [axes[1]]]

    for i, filename in enumerate(sample_files):
        raw_path       = os.path.join(raw_folder,       filename)
        processed_path = os.path.join(processed_folder, filename)

        if not os.path.exists(processed_path):
            print(f"  ✗ Missing processed file: {filename}")
            continue

        raw_img       = cv2.cvtColor(cv2.imread(raw_path),       cv2.COLOR_BGR2RGB)
        processed_img = cv2.cvtColor(cv2.imread(processed_path), cv2.COLOR_BGR2RGB)

        # Top row: raw
        axes[0][i].imshow(raw_img)
        axes[0][i].set_title(filename, fontsize=7)
        axes[0][i].axis('off')


        axes[1][i].imshow(processed_img)
        axes[1][i].set_title("Processed (224×224)", fontsize=7)
        axes[1][i].axis('off')

    plt.tight_layout()

    safe_title  = title.replace("/", "_").replace(" ", "_")
    report_path = os.path.join(PROJECT_ROOT, "reports", f"check_{safe_title}.png")
    plt.savefig(report_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: reports/check_{safe_title}.png")
    plt.show()
    plt.close()  # Free memory — important when looping many classes



print("\n" + "=" * 50)
print("  Checking My_Garden classes")
print("=" * 50)

garden_classes = ["Rose_Disease", "Hibiscus_Healthy", "Parijat_Healthy"]

for class_name in garden_classes:
    raw_path       = os.path.join(RAW_DIR,       "My_Garden", class_name)
    processed_path = os.path.join(PROCESSED_DIR, "My_Garden", class_name)
    print(f"\n→ {class_name}")
    check_category(raw_path, processed_path, f"My_Garden / {class_name}")



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