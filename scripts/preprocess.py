# scripts/preprocess.py
#
# FloraGuard — Phase 1: Preprocessing Pipeline
#
# What this script does:
#   1. Reads every raw image from raw_data/
#   2. Converts to HSV and creates a leaf mask
#   3. Cleans the mask with morphological operations
#   4. Crops tightly around the leaf (ROI)
#   5. Saves the cleaned image to processed_data/
#
# Run this script ONCE before training. Every image the model sees
# will be a product of this pipeline.

import cv2          # OpenCV — our computer vision library
import numpy as np  # NumPy — for array/matrix operations on images
import os           # os — for navigating folders and file paths
import sys          # sys — for clean error exits
from tqdm import tqdm  # Progress bar for long-running loops


# ===========================================================
# SECTION 1: PATH CONFIGURATION
# ===========================================================
# We use os.path so this works on Windows, Mac, and Linux.
# __file__ refers to this script itself.
# We go one level up (..) from scripts/ to reach the project root.

# Project root = FloraGuard_Project/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Where raw images live
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "raw_data")

# Where processed images will be saved
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")

# Final size of every output image (width x height in pixels)
# 224x224 is the standard input size for MobileNetV2, EfficientNetB0, ResNet50
# By resizing here, we ensure all three models get identical input dimensions
OUTPUT_SIZE = (224, 224)


# ===========================================================
# SECTION 2: HSV THRESHOLD CONFIGURATION
# ===========================================================
# These define what "green leaf" looks like in HSV space.
# H range 25–95 covers yellow-green to deep green (catches even
# slightly yellowed or diseased leaves, not just perfect green).
# S range 30–255 ensures some color richness (filters out gray backgrounds).
# V range 30–255 avoids pure black shadows.
#
# These are ADAPTIVE — you can tune them if results are poor.
# Lower H_MIN to catch more yellow. Raise S_MIN to reject pale backgrounds.

HSV_LOWER = np.array([25, 30, 30])   # Lower bound: [H_min, S_min, V_min]
HSV_UPPER = np.array([95, 255, 255]) # Upper bound: [H_max, S_max, V_max]

# Morphological kernel size — the "brush size" for erosion and dilation
# A 5x5 kernel is a good starting point. If mask has too much noise, increase to 7x7.
MORPH_KERNEL = np.ones((5, 5), np.uint8)


# ===========================================================
# SECTION 3: CORE PREPROCESSING FUNCTION
# ===========================================================

def preprocess_image(image_path, output_path, hsv_lower=HSV_LOWER, hsv_upper=HSV_UPPER):
    """
    Takes one raw image, runs the full preprocessing pipeline,
    and saves the cleaned result.

    Parameters:
        image_path  (str): Full path to the raw input image
        output_path (str): Full path where the processed image will be saved
        hsv_lower (np.array): Lower HSV bound for leaf color detection
        hsv_upper (np.array): Upper HSV bound for leaf color detection

    Returns:
        True if successful, False if something went wrong
    """

    # --- Step A: Load the image ---
    # cv2.imread loads image as a NumPy array of shape (height, width, 3)
    # The 3 channels are BGR (Blue, Green, Red) — note: OpenCV uses BGR not RGB!
    image_bgr = cv2.imread(image_path)

    # Check if image loaded correctly (returns None if file not found or corrupt)
    if image_bgr is None:
        print(f"  ✗ Could not load: {image_path}")
        return False

    # --- Step B: Convert BGR → HSV ---
    # cv2.cvtColor is the color space conversion function
    # cv2.COLOR_BGR2HSV tells it which conversion to perform
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # --- Step C: Create the leaf mask ---
    # cv2.inRange creates a binary (black/white) mask:
    #   - White (255) where pixel values fall WITHIN [hsv_lower, hsv_upper]
    #   - Black (0) everywhere else
    # This is our "find the green leaf" operation
    mask = cv2.inRange(image_hsv, hsv_lower, hsv_upper)

    # --- Step D: Clean the mask with morphological operations ---
    # MORPH_OPEN = erosion followed by dilation
    # This removes small white speckles (background noise) while
    # preserving the main leaf blob shape
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)

    # MORPH_CLOSE = dilation followed by erosion
    # This fills small black holes INSIDE the leaf blob
    # (happens when part of the leaf reflects light and appears non-green)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, MORPH_KERNEL)

    # --- Step E: Find contours in the cleaned mask ---
    # A contour is a list of (x, y) coordinates that trace the boundary of a white blob
    # cv2.findContours returns a list of all contours found
    # RETR_EXTERNAL = only find outermost contours (we don't want contours within contours)
    # CHAIN_APPROX_SIMPLE = compress the contour points (saves memory)
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours found, the masking failed — no leaf detected
    if len(contours) == 0:
        print(f"  ✗ No leaf detected in: {os.path.basename(image_path)}")
        # We still save the original resized image so the dataset isn't missing entries
        fallback = cv2.resize(image_bgr, OUTPUT_SIZE)
        cv2.imwrite(output_path, fallback)
        return False

    # --- Step F: Find the LARGEST contour ---
    # There might be multiple blobs (a leaf + a small green speck in background)
    # We assume the largest blob is the actual leaf
    # cv2.contourArea returns the area (in pixels) of a contour
    largest_contour = max(contours, key=cv2.contourArea)

    # --- Step G: Get bounding box of the largest contour ---
    # cv2.boundingRect returns (x, y, width, height) of the smallest
    # rectangle that completely contains the contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # --- Step H: Add padding around the bounding box ---
    # Without padding, the crop cuts right at the leaf edge.
    # A small 10-pixel padding gives the model a little context around the leaf.
    padding = 10
    x1 = max(0, x - padding)           # max(0,...) prevents going out of image bounds
    y1 = max(0, y - padding)
    x2 = min(image_bgr.shape[1], x + w + padding)  # shape[1] = image width
    y2 = min(image_bgr.shape[0], y + h + padding)  # shape[0] = image height

    # --- Step I: Crop the image using the bounding box ---
    # NumPy array slicing: image[y1:y2, x1:x2] cuts a rectangular region
    cropped = image_bgr[y1:y2, x1:x2]

    # Safety check — if crop is somehow empty, fall back to full image
    if cropped.size == 0:
        print(f"  ✗ Empty crop result for: {os.path.basename(image_path)}")
        cropped = image_bgr

    # --- Step J: Resize to standard output size ---
    # All images must be the same size before entering the neural network
    # cv2.INTER_AREA is the best interpolation method for shrinking images
    resized = cv2.resize(cropped, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)

    # --- Step K: Save the processed image ---
    cv2.imwrite(output_path, resized)
    return True


# ===========================================================
# SECTION 4: BATCH PROCESSING FUNCTION
# ===========================================================

def process_folder(input_folder, output_folder):
    """
    Processes all images in a folder and saves results to output_folder.
    Preserves the subfolder structure exactly.

    For example:
        input:  raw_data/My_Garden/Rose_Disease/Rose_001.jpg
        output: processed_data/My_Garden/Rose_Disease/Rose_001.jpg
    """

    success_count = 0
    fail_count = 0

    # os.walk traverses ALL subfolders recursively
    # root = current folder path
    # dirs = list of subfolders in root
    # files = list of files in root
    for root, dirs, files in os.walk(input_folder):

        # Filter to only image files (case-insensitive)
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            continue  # Skip folders with no images

        # Recreate the same subfolder structure in processed_data/
        # os.path.relpath gives the relative path from input_folder to root
        # e.g. if root = raw_data/My_Garden/Rose_Disease
        # then relative_path = My_Garden/Rose_Disease
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

# ===========================================================
# SECTION 4B: PLANTVILLAGE BATCH PROCESSING FUNCTION
# ===========================================================

def process_plantvillage(input_folder, output_folder):
    """
    Processes all PlantVillage images with a progress bar.
    
    PlantVillage structure:
        input_folder/
            Potato___Early_blight/   ← each subfolder is one disease class
                image001.JPG
                image002.JPG
                ...
    
    We preserve this structure in processed_data/PlantVillage/.
    
    Key difference from process_folder():
        - Uses tqdm progress bar (17k images need visual feedback)
        - Counts per-class statistics so you know which classes had issues
        - Prints a final summary table
    
    Parameters:
        input_folder  (str): Path to raw_data/PlantVillage/
        output_folder (str): Path to processed_data/PlantVillage/
    """

    # --- Step 1: Discover all class folders ---
    # os.listdir returns all items (files AND folders) in a directory
    # We filter with os.path.isdir to keep only folders (class names)
    # sorted() gives us alphabetical order — consistent across all runs
    
    class_folders = sorted([
        item for item in os.listdir(input_folder)
        if os.path.isdir(os.path.join(input_folder, item))
    ])
    
    # Safety check — if no class folders found, something is wrong
    if len(class_folders) == 0:
        print(f"✗ ERROR: No subfolders found in {input_folder}")
        print("  Make sure you copied the PlantVillage class folders correctly.")
        return

    print(f"\n📊 Found {len(class_folders)} class folders:")
    for folder in class_folders:
        print(f"   • {folder}")

    # --- Step 2: Build a complete list of all (input_path, output_path) pairs ---
    # We build the full list FIRST before processing.
    # Why? Because tqdm needs to know the total count upfront to show a percentage.
    # If we discovered files on the fly, tqdm wouldn't know "17000 total."
    
    all_tasks = []  # Each item = (input_image_path, output_image_path, class_name)

    for class_name in class_folders:
        
        # Full path to this class's raw image folder
        class_input_dir = os.path.join(input_folder, class_name)
        
        # Full path to where this class's processed images will go
        class_output_dir = os.path.join(output_folder, class_name)
        
        # Create the output class folder if it doesn't exist
        # exist_ok=True = don't crash if it already exists
        os.makedirs(class_output_dir, exist_ok=True)

        # Get all image files in this class folder
        # os.listdir gives filenames only (not full paths)
        # We filter by extension and build full paths with os.path.join
        image_files = [
            f for f in os.listdir(class_input_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        # Add each image as a task tuple to our master list
        for filename in image_files:
            input_path  = os.path.join(class_input_dir,  filename)
            output_path = os.path.join(class_output_dir, filename)
            all_tasks.append((input_path, output_path, class_name))

    # Report total count before starting
    total_images = len(all_tasks)
    print(f"\n🌿 Total images to process: {total_images}")

    # --- Step 3: Process every image with a progress bar ---
    # tqdm(all_tasks, ...) wraps our list
    # Every iteration yields one (input_path, output_path, class_name) tuple
    # The progress bar updates automatically after each iteration
    
    # We'll also track per-class success/fail counts using a dictionary
    # defaultdict(lambda: {...}) creates a new stats dict automatically
    # for any new class_name key we access — no KeyError
    from collections import defaultdict
    stats = defaultdict(lambda: {"success": 0, "fail": 0})

    with tqdm(
        total=total_images,
        desc="Processing PlantVillage",  # Label shown on the left of the bar
        unit="img",                       # Unit shown on the right (e.g. "30.2 img/s")
        dynamic_ncols=True                # Adjusts bar width to terminal size
    ) as progress_bar:

        for input_path, output_path, class_name in all_tasks:

            # Call our existing preprocess_image() function on each image
            # This is why we built it as a reusable function earlier —
            # we can call it from anywhere without rewriting the logic
            result = preprocess_image(input_path, output_path)

            # Update per-class stats
            if result:
                stats[class_name]["success"] += 1
            else:
                stats[class_name]["fail"] += 1

            # Advance the progress bar by 1 step
            progress_bar.update(1)

    # --- Step 4: Print per-class summary table ---
    # This tells you which classes had the most failures
    # (usually very diseased leaves that are mostly brown/yellow — outside our HSV range)
    
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
        # f-string formatting: :<40 = left-align in 40 chars, :>6 = right-align in 6 chars
        print(f"  {class_name:<40} {s:>6} {f:>6}")

    print("-" * 60)
    print(f"  {'TOTAL':<40} {total_success:>6} {total_fail:>6}")
    print("=" * 60)

    # Calculate and show failure rate
    fail_rate = (total_fail / total_images) * 100 if total_images > 0 else 0
    print(f"\n  Failure rate: {fail_rate:.1f}%")

    if fail_rate < 5:
        print("  ✅ Excellent — pipeline handled PlantVillage well.")
    elif fail_rate < 15:
        print("  ⚠️  Acceptable — some heavily diseased leaves fell back to resize-only.")
    else:
        print("  ⚠️  High failure rate — HSV thresholds may need tuning for PlantVillage.")

# ===========================================================
# SECTION 5: MAIN EXECUTION
# ===========================================================
# This block only runs when you execute this script directly.
# It does NOT run if another script imports this file as a module.
# That's what `if __name__ == "__main__":` means.

if __name__ == "__main__":

    print("=" * 60)
    print("  FloraGuard Preprocessing Pipeline")
    print("=" * 60)

    # -------------------------------------------------------
    # Phase A: Process My_Garden photos
    # Always run this first — it's fast (45 images) and
    # confirms the pipeline is working before the long run.
    # -------------------------------------------------------
    
    garden_input  = os.path.join(RAW_DATA_DIR,       "My_Garden")
    garden_output = os.path.join(PROCESSED_DATA_DIR, "My_Garden")

    print(f"\n🌿 Phase A: Processing My_Garden photos...")

    if not os.path.exists(garden_input):
        print(f"✗ ERROR: Could not find {garden_input}")
        sys.exit(1)

    success, fail = process_folder(garden_input, garden_output)
    print(f"\n  My_Garden: {success} processed ✓  {fail} failed ✗")

    # -------------------------------------------------------
    # Phase B: Process PlantVillage dataset
    # This is the long one — ~17,000 images with progress bar.
    # Estimated time: 10-20 minutes depending on your machine.
    # -------------------------------------------------------
    
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