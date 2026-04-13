import cv2          # OpenCV — our computer vision library
import numpy as np  # NumPy — for array/matrix operations on images
import os           # os — for navigating folders and file paths
import sys          # sys — for clean error exits
from tqdm import tqdm  # Progress bar for long-running loops



PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "raw_data")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
OUTPUT_SIZE = (224, 224)

HSV_LOWER = np.array([25, 30, 30])   
HSV_UPPER = np.array([95, 255, 255])

MORPH_KERNEL = np.ones((5, 5), np.uint8)


def preprocess_image(image_path, output_path, hsv_lower=HSV_LOWER, hsv_upper=HSV_UPPER):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"  ✗ Could not load: {image_path}")
        return False
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image_hsv, hsv_lower, hsv_upper)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, MORPH_KERNEL)
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print(f"  ✗ No leaf detected in: {os.path.basename(image_path)}")
        fallback = cv2.resize(image_bgr, OUTPUT_SIZE)
        cv2.imwrite(output_path, fallback)
        return False
    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)

    padding = 10
    x1 = max(0, x - padding)           
    y1 = max(0, y - padding)
    x2 = min(image_bgr.shape[1], x + w + padding)  
    y2 = min(image_bgr.shape[0], y + h + padding)  
    cropped = image_bgr[y1:y2, x1:x2]

    if cropped.size == 0:
        print(f"  ✗ Empty crop result for: {os.path.basename(image_path)}")
        cropped = image_bgr

    resized = cv2.resize(cropped, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)

    cv2.imwrite(output_path, resized)
    return True


def process_folder(input_folder, output_folder):

    success_count = 0
    fail_count = 0

    for root, dirs, files in os.walk(input_folder):

        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            continue  
        relative_path = os.path.relpath(root, input_folder)
        output_subdir = os.path.join(output_folder, relative_path)
        os.makedirs(output_subdir, exist_ok=True)

        print(f"\n📁 Processing: {relative_path}")

        for filename in image_files:
            input_path = os.path.join(root, filename)
            output_path = os.path.join(output_subdir, filename)

            print(f"  → {filename}", end="  ")
            result = preprocess_image(input_path, output_path)

            if result:
                print("✓")
                success_count += 1
            else:
                fail_count += 1

    return success_count, fail_count

def process_plantvillage(input_folder, output_folder):    
    class_folders = sorted([
        item for item in os.listdir(input_folder)
        if os.path.isdir(os.path.join(input_folder, item))
    ])
    if len(class_folders) == 0:
        print(f"✗ ERROR: No subfolders found in {input_folder}")
        print("  Make sure you copied the PlantVillage class folders correctly.")
        return

    print(f"\n📊 Found {len(class_folders)} class folders:")
    for folder in class_folders:
        print(f"   • {folder}")
    
    all_tasks = []  

    for class_name in class_folders:
        class_input_dir = os.path.join(input_folder, class_name)
        class_output_dir = os.path.join(output_folder, class_name)
        
        os.makedirs(class_output_dir, exist_ok=True)

        
        image_files = [
            f for f in os.listdir(class_input_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        
        for filename in image_files:
            input_path  = os.path.join(class_input_dir,  filename)
            output_path = os.path.join(class_output_dir, filename)
            all_tasks.append((input_path, output_path, class_name))

    
    total_images = len(all_tasks)
    print(f"\n🌿 Total images to process: {total_images}")


    from collections import defaultdict
    stats = defaultdict(lambda: {"success": 0, "fail": 0})

    with tqdm(
        total=total_images,
        desc="Processing PlantVillage",  
        unit="img",                       
        dynamic_ncols=True                
    ) as progress_bar:

        for input_path, output_path, class_name in all_tasks:

            result = preprocess_image(input_path, output_path)

            if result:
                stats[class_name]["success"] += 1
            else:
                stats[class_name]["fail"] += 1

            progress_bar.update(1)

    
    print("\n" + "=" * 60)
    print(f"  PlantVillage Processing Summary")
    print("=" * 60)
    print(f"  {'Class':<40} {'✓':>6} {'✗':>6}")
    print("-" * 60)

    total_success = 0
    total_fail    = 0

    for class_name in sorted(stats.keys()):
        s = stats[class_name]["success"]
        f = stats[class_name]["fail"]
        total_success += s
        total_fail    += f
        print(f"  {class_name:<40} {s:>6} {f:>6}")

    print("-" * 60)
    print(f"  {'TOTAL':<40} {total_success:>6} {total_fail:>6}")
    print("=" * 60)

    fail_rate = (total_fail / total_images) * 100 if total_images > 0 else 0
    print(f"\n  Failure rate: {fail_rate:.1f}%")

    if fail_rate < 5:
        print("  ✅ Excellent — pipeline handled PlantVillage well.")
    elif fail_rate < 15:
        print("  ⚠️  Acceptable — some heavily diseased leaves fell back to resize-only.")
    else:
        print("  ⚠️  High failure rate — HSV thresholds may need tuning for PlantVillage.")


if __name__ == "__main__":

    print("=" * 60)
    print("  FloraGuard Preprocessing Pipeline")
    print("=" * 60)

    
    garden_input  = os.path.join(RAW_DATA_DIR,       "My_Garden")
    garden_output = os.path.join(PROCESSED_DATA_DIR, "My_Garden")

    print(f"\n🌿 Phase A: Processing My_Garden photos...")

    if not os.path.exists(garden_input):
        print(f"✗ ERROR: Could not find {garden_input}")
        sys.exit(1)

    success, fail = process_folder(garden_input, garden_output)
    print(f"\n  My_Garden: {success} processed ✓  {fail} failed ✗")
    
    plantvillage_input  = os.path.join(RAW_DATA_DIR,       "PlantVillage")
    plantvillage_output = os.path.join(PROCESSED_DATA_DIR, "PlantVillage")

    print(f"\n🌿 Phase B: Processing PlantVillage dataset...")

    if not os.path.exists(plantvillage_input):
        print(f"✗ ERROR: Could not find {plantvillage_input}")
        print("  Make sure PlantVillage folders are in raw_data/PlantVillage/")
        sys.exit(1)

    process_plantvillage(plantvillage_input, plantvillage_output)

    print("\n✅ Full preprocessing pipeline complete.")
    print(f"   All cleaned images are in: {PROCESSED_DATA_DIR}")