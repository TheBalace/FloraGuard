"""
FloraGuard — Phase 4: Streamlit App
Plant disease diagnosis with OpenCV preprocessing + EfficientNetB0 + Grad-CAM++
"""

import os
import cv2
import numpy as np
import streamlit as st
import matplotlib.cm as cm
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, ResNet50
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess



BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
NPY_PATH = os.path.join(BASE_DIR, '..', 'models', 'floraguard_weights.npy')

IMAGE_SIZE  = (224, 224)
NUM_CLASSES = 10

CLASS_NAMES = [
    'Hibiscus_Healthy',
    'Parijat_Healthy',
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Rose_Disease',
    'Tomato_Bacterial_spot',
    'Tomato_healthy',
]

HSV_LOWER    = np.array([25, 30, 30],   dtype=np.uint8)
HSV_UPPER    = np.array([95, 255, 255], dtype=np.uint8)
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
PADDING      = 10



def build_model(backbone_name: str) -> tf.keras.Model:
    backbone_config = {
        'MobileNetV2':    (MobileNetV2,    tf.keras.applications.mobilenet_v2.preprocess_input),
        'EfficientNetB0': (EfficientNetB0, efficientnet_preprocess),
        'ResNet50':       (ResNet50,        tf.keras.applications.resnet50.preprocess_input),
    }
    BackboneClass, preprocess_fn = backbone_config[backbone_name]

    inputs = tf.keras.Input(shape=(224, 224, 3))

    x = Lambda(
        lambda img: preprocess_fn(img),
        name='backbone_preprocessing'
    )(inputs)

    backbone = BackboneClass(
        include_top=False,
        weights='imagenet',
        input_tensor=x        
    )
    backbone.trainable = False

    x = backbone.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


# NEW
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'efficientnet_best.keras')

@st.cache_resource
def load_efficientnet() -> tf.keras.Model:
    model = build_model('EfficientNetB0')
    model.load_weights(MODEL_PATH)
    return model

def preprocess_uploaded_image(pil_image: Image.Image) -> np.ndarray | None:
    img_rgb = np.array(pil_image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  MORPH_KERNEL)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest  = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    H, W     = img_bgr.shape[:2]
    x1, y1   = max(0, x - PADDING),     max(0, y - PADDING)
    x2, y2   = min(W, x + w + PADDING), min(H, y + h + PADDING)

    cropped = img_bgr[y1:y2, x1:x2]
    resized = cv2.resize(cropped, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)   # Return RGB


def predict(model: tf.keras.Model, preprocessed_rgb: np.ndarray):
    img_float = efficientnet_preprocess(preprocessed_rgb.astype('float32'))
    img_batch = np.expand_dims(img_float, axis=0)
    probs     = model.predict(img_batch, verbose=0)[0]
    class_idx = int(np.argmax(probs))
    return CLASS_NAMES[class_idx], float(probs[class_idx]), probs, class_idx



def generate_gradcam(model: tf.keras.Model,
                     preprocessed_rgb: np.ndarray,
                     class_idx: int) -> np.ndarray:

    conv_submodel = tf.keras.Model(
    inputs  = model.inputs,
    outputs = model.get_layer('top_conv').output
)

    img_float = efficientnet_preprocess(preprocessed_rgb.astype('float32'))
    img_batch = tf.cast(np.expand_dims(img_float, axis=0), tf.float32)

    with tf.GradientTape() as tape:

        conv_outputs = conv_submodel(img_batch, training=False)
        tape.watch(conv_outputs)   


        x = model.get_layer('top_bn')(conv_outputs, training=False)
        x = model.get_layer('top_activation')(x)


        x = model.get_layer('global_average_pooling2d')(x)
        x = model.get_layer('batch_normalization')(x, training=False)
        x = model.get_layer('dense')(x)
        x = model.get_layer('dropout')(x, training=False)
        predictions = model.get_layer('dense_1')(x)

        loss = predictions[:, class_idx]

    grads       = tape.gradient(loss, conv_outputs)       
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)) 

    conv_outputs = conv_outputs[0]                                     
    heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]        
    heatmap      = tf.squeeze(heatmap)                                 
    heatmap      = tf.maximum(heatmap, 0)
    heatmap      = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap      = heatmap.numpy()

    heatmap_resized = cv2.resize(heatmap, IMAGE_SIZE)
    heatmap_colored = (cm.jet(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
    overlay         = cv2.addWeighted(preprocessed_rgb, 0.6, heatmap_colored, 0.4, 0)

    return overlay


def main():
    st.set_page_config(page_title='FloraGuard', page_icon='🌿', layout='wide')

    st.title('🌿 FloraGuard')
    st.subheader('Plant Disease Diagnosis with Explainable AI')
    st.markdown(
        'Upload a photo of a plant leaf. FloraGuard removes the background, '
        'identifies the disease, and shows **where** the model is looking via Grad-CAM++.'
    )
    st.divider()

    with st.spinner('Loading EfficientNetB0 model...'):
        model = load_efficientnet()

    uploaded_file = st.file_uploader('Upload a plant photo', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is None:
        st.info('👆 Upload an image above to get started.')
        return

    pil_image = Image.open(uploaded_file)

    with st.spinner('Running OpenCV preprocessing pipeline...'):
        preprocessed = preprocess_uploaded_image(pil_image)

    if preprocessed is None:
        st.error(
            '⚠️ No leaf detected. The HSV pipeline could not find a green region. '
            'Try a clearer photo with a visible leaf against a contrasting background.'
        )
        return

    with st.spinner('Running inference...'):
        class_name, confidence, all_probs, class_idx = predict(model, preprocessed)

    with st.spinner('Generating Grad-CAM++ attention map...'):
        overlay = generate_gradcam(model, preprocessed, class_idx)

    display_name = class_name.replace('_', ' ')
    if 'healthy' in class_name.lower() or 'Healthy' in class_name:
        st.success(f'✅  **{display_name}** — Confidence: **{confidence * 100:.1f}%**')
    else:
        st.warning(f'⚠️  **{display_name}** — Confidence: **{confidence * 100:.1f}%**')

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('#### 📷 Original Upload')
        st.image(pil_image, use_column_width=True)

    with col2:
        st.markdown('#### 🔬 After OpenCV Preprocessing')
        st.image(preprocessed, use_column_width=True)

    with col3:
        st.markdown('#### 🔥 Grad-CAM++ Attention Map')
        st.image(overlay, use_column_width=True)
        st.caption('Red = high attention · Blue = low attention')

    st.divider()

    with st.expander('📊 Full Confidence Breakdown (all 10 classes)'):
        sorted_preds = sorted(zip(CLASS_NAMES, all_probs), key=lambda x: -x[1])
        for name, prob in sorted_preds:
            st.progress(float(prob), text=f'{name.replace("_", " ")}: {prob * 100:.2f}%')


if __name__ == '__main__':
    main()