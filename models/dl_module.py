"""
dl_module.py - Deep Learning Module
Image classification using pretrained MobileNetV2/ResNet50 + OpenCV object detection
"""

import streamlit as st
import numpy as np
import cv2
import io
import warnings
warnings.filterwarnings("ignore")

from PIL import Image

# ─── Lazy imports ────────────────────────────────────────────────────────────

def _load_tf_model(model_name):
    """Load a Keras pretrained model."""
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mn_pre, decode_predictions as mn_dec
    from tensorflow.keras.applications.resnet50  import preprocess_input as rn_pre, decode_predictions as rn_dec
    from tensorflow.keras.applications.vgg16     import preprocess_input as vg_pre, decode_predictions as vg_dec

    models_map = {
        "MobileNetV2": (MobileNetV2, mn_pre, mn_dec, (224, 224)),
        "ResNet50":    (ResNet50,    rn_pre, rn_dec, (224, 224)),
        "VGG16":       (VGG16,       vg_pre, vg_dec, (224, 224)),
    }
    ModelClass, preprocess, decode, size = models_map[model_name]
    model = ModelClass(weights="imagenet")
    return model, preprocess, decode, size


def _classify_image_tf(image_pil, model_name):
    """Classify an image using TF/Keras pretrained model."""
    import numpy as np
    from tensorflow.keras.preprocessing.image import img_to_array

    model, preprocess, decode, (h, w) = _load_tf_model(model_name)
    img = image_pil.convert("RGB").resize((w, h))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess(arr)
    preds = model.predict(arr, verbose=0)
    top = decode(preds, top=5)[0]
    results = [{"Rank": i+1, "Label": label.replace("_", " ").title(),
                "Confidence": f"{prob*100:.2f}%", "Score": round(prob, 4)}
               for i, (_, label, prob) in enumerate(top)]
    return results


def _classify_image_torch(image_pil, model_name):
    """Classify an image using PyTorch pretrained model."""
    import torch
    import torchvision.transforms as T
    import torchvision.models as models_tv
    import json
    import urllib.request

    # Load imagenet class labels
    LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    try:
        with urllib.request.urlopen(LABELS_URL, timeout=5) as r:
            class_labels = json.load(r)
    except Exception:
        class_labels = [str(i) for i in range(1000)]

    torch_models = {
        "MobileNetV2": models_tv.mobilenet_v2,
        "ResNet50":    models_tv.resnet50,
    }
    model_fn = torch_models.get(model_name, models_tv.mobilenet_v2)
    model = model_fn(pretrained=True)
    model.eval()

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = image_pil.convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)

    top_probs, top_idxs = torch.topk(probs, 5)
    results = []
    for i, (prob, idx) in enumerate(zip(top_probs, top_idxs)):
        label = class_labels[idx.item()] if idx.item() < len(class_labels) else str(idx.item())
        results.append({
            "Rank": i+1,
            "Label": label.replace("_", " ").title(),
            "Confidence": f"{prob.item()*100:.2f}%",
            "Score": round(prob.item(), 4),
        })
    return results


def detect_edges_opencv(image_pil):
    """Apply Canny edge detection using OpenCV."""
    img_array = np.array(image_pil.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    return edges


def detect_faces_opencv(image_pil):
    """Detect faces using Haar Cascade classifier."""
    img_array = np.array(image_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    result_img = img_array.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 200, 255), 2)
        cv2.putText(result_img, "Face", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    return result_img, len(faces)


def apply_image_filters(image_pil):
    """Apply various OpenCV image processing filters and return dict of results."""
    img = np.array(image_pil.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    contours_img = img.copy()
    contours, _ = cv2.findContours(
        cv2.Canny(gray, 50, 150), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(contours_img, contours, -1, (0, 255, 120), 1)

    return {
        "Grayscale": gray,
        "Blurred": blurred,
        "Sharpened": sharpened,
        "Threshold": thresh,
        "Contours": contours_img,
    }


# ─── Streamlit UI ─────────────────────────────────────────────────────────────

def render_dl_module():
    st.header("🧠 Deep Learning Module")
    st.markdown("Upload an image to classify it with pretrained CNNs or run OpenCV computer vision pipelines.")

    uploaded = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="dl_upload")

    if uploaded is None:
        st.info("👆 Upload an image (JPG or PNG) to begin. Try uploading a photo of an animal, vehicle, or everyday object.")
        return

    image_pil = Image.open(uploaded)
    st.image(image_pil, caption="Uploaded Image", use_column_width=True)

    tabs = st.tabs(["🏷️ Image Classification", "👁️ OpenCV Analysis", "🎨 Image Filters"])

    # ── Tab 1: Classification ─────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Image Classification (ImageNet)")

        backend = st.radio("Choose Backend", ["TensorFlow/Keras", "PyTorch"], horizontal=True)
        if backend == "TensorFlow/Keras":
            model_choice = st.selectbox("Model", ["MobileNetV2", "ResNet50", "VGG16"])
        else:
            model_choice = st.selectbox("Model", ["MobileNetV2", "ResNet50"])

        if st.button("🔍 Classify Image", type="primary"):
            with st.spinner(f"Running {model_choice} inference..."):
                try:
                    if backend == "TensorFlow/Keras":
                        results = _classify_image_tf(image_pil, model_choice)
                    else:
                        results = _classify_image_torch(image_pil, model_choice)

                    import pandas as pd
                    import matplotlib.pyplot as plt

                    st.success(f"✅ Top prediction: **{results[0]['Label']}** ({results[0]['Confidence']})")
                    st.subheader("Top 5 Predictions")
                    df_preds = pd.DataFrame(results)
                    st.dataframe(df_preds, use_container_width=True)

                    # Bar chart of confidences
                    fig, ax = plt.subplots(figsize=(8, 4))
                    labels = [r["Label"][:30] for r in results]
                    scores = [r["Score"] for r in results]
                    colors = ["#0ea5e9" if i == 0 else "#334155" for i in range(len(scores))]
                    bars = ax.barh(labels[::-1], scores[::-1], color=colors[::-1])
                    ax.set_xlabel("Confidence Score")
                    ax.set_title("Top 5 Predictions")
                    ax.set_xlim(0, max(scores) * 1.2)
                    for bar, score in zip(bars, scores[::-1]):
                        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                                f"{score*100:.1f}%", va="center", fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig)

                except Exception:
                    st.error("Classification failed. Check the image and model setup, then try again.")
                    st.info("Make sure TensorFlow or PyTorch is installed. Run: `pip install tensorflow` or `pip install torch torchvision`")

    # ── Tab 2: OpenCV Analysis ────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("OpenCV Computer Vision")

        cv_task = st.selectbox("Select Analysis", ["Edge Detection", "Face Detection"])

        if st.button("▶ Run OpenCV Analysis", type="primary"):
            with st.spinner("Processing with OpenCV..."):
                if cv_task == "Edge Detection":
                    edges = detect_edges_opencv(image_pil)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image_pil, caption="Original", use_column_width=True)
                    with col2:
                        st.image(edges, caption="Canny Edge Detection", use_column_width=True, clamp=True)
                    st.info(f"Detected approximately **{np.sum(edges > 0):,}** edge pixels.")

                elif cv_task == "Face Detection":
                    result_img, face_count = detect_faces_opencv(image_pil)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image_pil, caption="Original", use_column_width=True)
                    with col2:
                        st.image(result_img, caption="Face Detection", use_column_width=True)
                    if face_count > 0:
                        st.success(f"✅ Detected **{face_count}** face(s).")
                    else:
                        st.warning("No faces detected. Try a clear portrait photo.")

    # ── Tab 3: Image Filters ──────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("OpenCV Image Processing Filters")
        if st.button("🎨 Apply All Filters", type="primary"):
            with st.spinner("Applying filters..."):
                filters = apply_image_filters(image_pil)
                cols = st.columns(3)
                for i, (name, img) in enumerate(filters.items()):
                    with cols[i % 3]:
                        if len(img.shape) == 2:
                            st.image(img, caption=name, use_column_width=True, clamp=True)
                        else:
                            st.image(img, caption=name, use_column_width=True)
