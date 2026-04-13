import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR   = os.path.join(PROJECT_ROOT, "reports")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed_data")
FLAT_DIR = os.path.join(PROJECT_ROOT, "flat_data")

IMAGE_SIZE  = (224, 224)
BATCH_SIZE  = 32


def build_flat_directory():
    import shutil

    if os.path.exists(FLAT_DIR):
        shutil.rmtree(FLAT_DIR)
    os.makedirs(FLAT_DIR)

    print("📁 Building flat data directory...")

    class_counts = {}

    for root, dirs, files in os.walk(PROCESSED_DIR):
        image_files = [
            f for f in files
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        if not image_files:
            continue

        class_name    = os.path.basename(root)
        class_out_dir = os.path.join(FLAT_DIR, class_name)
        os.makedirs(class_out_dir, exist_ok=True)

        for filename in image_files:
            src = os.path.join(root, filename)
            dst = os.path.join(class_out_dir, filename)
            shutil.copy2(src, dst)

        class_counts[class_name] = len(image_files)
        print(f"  ✓ {class_name:<45} {len(image_files):>5} images")

    print(f"\n  Total: {sum(class_counts.values())} images, "
          f"{len(class_counts)} classes\n")

    return sorted(class_counts.keys())


def get_test_generator(flat_dir):


    test_datagen = ImageDataGenerator(validation_split=0.15)

    test_generator = test_datagen.flow_from_directory(
        flat_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',   
        shuffle=False,        
        seed=42                
    )


    index_to_class = {v: k for k, v in test_generator.class_indices.items()}
    class_names    = [index_to_class[i] for i in range(len(index_to_class))]

    print(f"📊 Test set: {test_generator.samples} images, "
          f"{len(class_names)} classes")

    return test_generator, class_names



def evaluate_model(model_path, model_name, test_generator, class_names):
    

    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name}")
    print(f"{'='*60}")


    if not os.path.exists(model_path):
        print(f"  ✗ Model file not found: {model_path}")
        print(f"    Make sure you downloaded it from Google Drive to models/")
        return None

    print(f"  Loading model from: {model_path}")
    model = keras.models.load_model(model_path, safe_mode=False)
    print(f"  ✓ Model loaded")


    test_generator.reset()

    print(f"  Running predictions on {test_generator.samples} test images...")


    predictions_prob = model.predict(
        test_generator,
        steps=np.ceil(test_generator.samples / BATCH_SIZE),
        verbose=1
    )


    predictions = np.argmax(predictions_prob, axis=1)


    true_labels = test_generator.classes


    accuracy = np.mean(predictions == true_labels)
    print(f"\n  Overall Test Accuracy: {accuracy*100:.2f}%")

    report = classification_report(
        true_labels,
        predictions,
        target_names=class_names,
        zero_division=0
    )
    print(f"\n  Classification Report:")
    print(f"  {'-'*58}")
    for line in report.split('\n'):
        print(f"  {line}")


    report_dict = classification_report(
        true_labels,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )


    cm = confusion_matrix(true_labels, predictions)


    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    ax.set_title(f'{model_name} — Confusion Matrix\n'
                 f'Test Accuracy: {accuracy*100:.2f}%',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)


    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    cm_path = os.path.join(REPORTS_DIR, f'confusion_matrix_{model_name}.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ Confusion matrix saved: reports/confusion_matrix_{model_name}.png")


    print(f"\n  Measuring inference speed...")

    dummy_batch = np.zeros((1, 224, 224, 3), dtype=np.float32)
    model.predict(dummy_batch, verbose=0)

    NUM_TIMING_RUNS = 10
    times = []

    for _ in range(NUM_TIMING_RUNS):
        test_batch = np.random.randint(
            0, 255,
            (BATCH_SIZE, 224, 224, 3)
        ).astype(np.float32)

        start = time.perf_counter()    # High precision timer
        model.predict(test_batch, verbose=0)
        end   = time.perf_counter()

        times.append(end - start)

    avg_batch_time = np.mean(times)           # Average seconds per batch
    fps            = BATCH_SIZE / avg_batch_time  # Images per second

    print(f"  Avg time per batch ({BATCH_SIZE} images): {avg_batch_time*1000:.1f} ms")
    print(f"  Inference speed: {fps:.1f} FPS")

    return {
        'model_name' : model_name,
        'accuracy'   : accuracy,
        'fps'        : fps,
        'report_dict': report_dict,
        'cm'         : cm
    }



def generate_comparison_table(results):
   

    print(f"\n{'='*60}")
    print(f"  Generating Comparison Table")
    print(f"{'='*60}")

    results = [r for r in results if r is not None]

    if not results:
        print("  ✗ No results to compare")
        return

    model_names = [r['model_name'] for r in results]
    accuracies  = [f"{r['accuracy']*100:.2f}%" for r in results]
    fps_values  = [f"{r['fps']:.1f}" for r in results]

    macro_f1s = [
        f"{r['report_dict']['macro avg']['f1-score']*100:.2f}%"
        for r in results
    ]

    weighted_f1s = [
        f"{r['report_dict']['weighted avg']['f1-score']*100:.2f}%"
        for r in results
    ]

    print(f"\n  {'Model':<20} {'Test Acc':>10} {'Macro F1':>10} "
          f"{'Weighted F1':>12} {'FPS':>8}")
    print(f"  {'-'*64}")

    for i, r in enumerate(results):
        print(f"  {model_names[i]:<20} {accuracies[i]:>10} "
              f"{macro_f1s[i]:>10} {weighted_f1s[i]:>12} "
              f"{fps_values[i]:>8}")

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')  

    table_data = [
        [model_names[i], accuracies[i], macro_f1s[i],
         weighted_f1s[i], fps_values[i]]
        for i in range(len(results))
    ]

    col_labels = ['Model', 'Test Accuracy', 'Macro F1',
                  'Weighted F1', 'FPS']

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.0)  

    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#2E86AB')  # Blue header
        table[0, j].set_text_props(color='white', fontweight='bold')

    for i in range(1, len(results) + 1):
        color = '#F0F4F8' if i % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(color)

    plt.title('FloraGuard — Ablation Study Results',
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    table_path = os.path.join(REPORTS_DIR, 'comparison_table.png')
    plt.savefig(table_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ Comparison table saved: reports/comparison_table.png")


if __name__ == "__main__":

    print("=" * 60)
    print("  FloraGuard — Model Evaluation Pipeline")
    print("=" * 60)

    class_names = build_flat_directory()

    test_generator, class_names = get_test_generator(FLAT_DIR)

    models_to_evaluate = [
        (os.path.join(MODELS_DIR, 'mobilenet_best.keras'),    'MobileNetV2'),
        (os.path.join(MODELS_DIR, 'efficientnet_best.keras'), 'EfficientNetB0'),
        (os.path.join(MODELS_DIR, 'resnet_best.keras'),       'ResNet50'),
    ]

    all_results = []

    for model_path, model_name in models_to_evaluate:
        try:
            result = evaluate_model(
                model_path,
                model_name,
                test_generator,
                class_names
            )
            all_results.append(result)
        except Exception as e:
            import traceback
            print(f"\n✗ ERROR evaluating {model_name}:")
            print(traceback.format_exc())
            all_results.append(None)

    generate_comparison_table(all_results)

    print(f"\n{'='*60}")
    print(f"  ✅ Evaluation complete.")
    print(f"  All outputs saved to: {REPORTS_DIR}")
    print(f"{'='*60}")