"""Image analysis helpers. Heavy model libraries load only when a model is used."""

from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw


MAX_ANALYSIS_SIDE = 1280


def resize_for_analysis(image, max_side=MAX_ANALYSIS_SIDE):
    """Return a copy sized for responsive CPU inference and image processing."""
    image = image.convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def _load_opencv():
    try:
        import cv2
    except (ImportError, OSError) as exc:
        raise RuntimeError("OpenCV is unavailable in this deployment.") from exc
    return cv2


def _load_torchvision():
    try:
        import torch
        import torchvision
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError("The image model runtime could not be loaded.") from exc
    torch.set_num_threads(1)
    return torch, torchvision


@lru_cache(maxsize=2)
def _load_classifier(model_name):
    torch, tv = _load_torchvision()
    options = {
        "MobileNetV2": (tv.models.mobilenet_v2, tv.models.MobileNet_V2_Weights.DEFAULT),
        "ResNet50": (tv.models.resnet50, tv.models.ResNet50_Weights.DEFAULT),
    }
    if model_name not in options:
        raise ValueError("Choose MobileNetV2 or ResNet50.")
    factory, weights = options[model_name]
    model = factory(weights=weights).eval()
    return model, weights


def _top_predictions(logits, weights, limit=5):
    torch, _ = _load_torchvision()
    probabilities = torch.softmax(logits[0], dim=0)
    scores, indices = torch.topk(probabilities, k=min(limit, probabilities.numel()))
    categories = weights.meta["categories"]
    return [
        {
            "Rank": rank,
            "Label": categories[index].replace("_", " ").title(),
            "Confidence": f"{score * 100:.2f}%",
            "Score": round(float(score), 5),
        }
        for rank, (score, index) in enumerate(zip(scores.tolist(), indices.tolist()), start=1)
    ]


def _prepare_classifier_input(image, weights):
    torch, _ = _load_torchvision()
    tensor = weights.transforms()(resize_for_analysis(image).convert("RGB"))
    return tensor.unsqueeze(0)


def classify_image(image, model_name="MobileNetV2"):
    torch, _ = _load_torchvision()
    model, weights = _load_classifier(model_name)
    tensor = _prepare_classifier_input(image, weights)
    with torch.inference_mode():
        logits = model(tensor)
    return _top_predictions(logits, weights)


def gradcam_image(image, model_name="MobileNetV2"):
    """Return original-sized heatmap overlay and its top ImageNet prediction."""
    torch, tv = _load_torchvision()
    model, weights = _load_classifier(model_name)
    last_conv = next(module for module in reversed(list(model.modules()))
                     if isinstance(module, torch.nn.Conv2d))
    captured = {}

    def capture(_module, _inputs, output):
        captured["features"] = output

    hook = last_conv.register_forward_hook(capture)
    try:
        tensor = _prepare_classifier_input(image, weights)
        logits = model(tensor)
        class_index = int(logits[0].argmax().item())
        features = captured["features"]
        gradients = torch.autograd.grad(logits[0, class_index], features)[0]
        channel_weights = gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((channel_weights * features).sum(dim=1, keepdim=True))
        heatmap = torch.nn.functional.interpolate(
            heatmap, size=(image.height, image.width), mode="bilinear", align_corners=False
        )[0, 0]
        heatmap = heatmap / heatmap.max().clamp(min=1e-8)
        heat = np.asarray(heatmap.detach().cpu().numpy() * 255, dtype=np.uint8)
        cv2 = _load_opencv()
        colored = cv2.cvtColor(cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
        base = np.asarray(image.convert("RGB"), dtype=np.float32)
        overlay = Image.fromarray(np.clip(base * 0.58 + colored.astype(np.float32) * 0.42, 0, 255).astype(np.uint8))
        prediction = _top_predictions(logits.detach(), weights, limit=1)[0]
        return overlay, prediction
    finally:
        hook.remove()


@lru_cache(maxsize=1)
def _load_object_detector():
    torch, tv = _load_torchvision()
    weights = tv.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    return tv.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights).eval(), weights


def detect_objects(image, score_threshold=0.45):
    torch, _ = _load_torchvision()
    model, weights = _load_object_detector()
    working = resize_for_analysis(image)
    tensor = weights.transforms()(working)
    with torch.inference_mode():
        output = model([tensor])[0]
    labels = weights.meta["categories"]
    draw = ImageDraw.Draw(working)
    found = []
    for box, label_id, score in zip(output["boxes"], output["labels"], output["scores"]):
        score = float(score)
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = [int(round(value)) for value in box.tolist()]
        label = labels[int(label_id)]
        draw.rectangle((x1, y1, x2, y2), outline="#f0b429", width=3)
        draw.text((x1 + 3, max(0, y1 - 16)), f"{label} {score:.0%}", fill="#f0b429")
        found.append({"Object": label, "Confidence": f"{score:.1%}"})
    return working, found


@lru_cache(maxsize=1)
def _load_segmentation_model():
    torch, tv = _load_torchvision()
    weights = tv.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    return tv.models.segmentation.deeplabv3_mobilenet_v3_large(weights=weights).eval(), weights


def segment_image(image):
    """Return a semantic segmentation overlay and area summary."""
    torch, _ = _load_torchvision()
    model, weights = _load_segmentation_model()
    working = resize_for_analysis(image, max_side=768)
    tensor = weights.transforms()(working)
    with torch.inference_mode():
        logits = model(tensor.unsqueeze(0))["out"]
        labels_map = logits.argmax(1)[0]
        labels_map = torch.nn.functional.interpolate(
            labels_map[None, None].float(), size=(working.height, working.width), mode="nearest"
        )[0, 0].to(torch.int64).cpu().numpy()
    categories = weights.meta["categories"]
    counts = np.bincount(labels_map.reshape(-1), minlength=len(categories))
    total = max(1, labels_map.size)
    areas = sorted(
        ((categories[i], count / total) for i, count in enumerate(counts) if count),
        key=lambda item: item[1], reverse=True,
    )[:8]
    # Stable palette keeps identical labels visually consistent across runs.
    palette = np.zeros((len(categories), 3), dtype=np.uint8)
    for i in range(len(categories)):
        palette[i] = ((37 * i + 70) % 220 + 25, (67 * i + 110) % 220 + 25, (97 * i + 30) % 220 + 25)
    color_mask = palette[labels_map]
    base = np.asarray(working, dtype=np.float32)
    overlay = Image.fromarray(np.clip(base * 0.52 + color_mask.astype(np.float32) * 0.48, 0, 255).astype(np.uint8))
    summary = [{"Region": name.replace("_", " ").title(), "Image area": f"{ratio:.1%}"} for name, ratio in areas]
    return overlay, summary


@lru_cache(maxsize=1)
def _load_face_cascade():
    cv2 = _load_opencv()
    data = getattr(cv2, "data", None)
    base = getattr(data, "haarcascades", None)
    if not base or not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError("This OpenCV build does not include the face detector.")
    path = base + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(path)
    if detector.empty():
        raise RuntimeError("The OpenCV face detection model could not be loaded.")
    return detector


def detect_faces(image):
    cv2 = _load_opencv()
    working = resize_for_analysis(image)
    rgb = np.asarray(working)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = _load_face_cascade().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    result = working.copy()
    draw = ImageDraw.Draw(result)
    for x, y, width, height in faces:
        draw.rectangle((int(x), int(y), int(x + width), int(y + height)), outline="#f0b429", width=3)
        draw.text((int(x) + 3, max(0, int(y) - 16)), "Face", fill="#f0b429")
    return result, len(faces)


def detect_edges(image, threshold1=50, threshold2=150):
    cv2 = _load_opencv()
    gray = cv2.cvtColor(np.asarray(resize_for_analysis(image)), cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blurred, int(threshold1), int(threshold2))


def apply_image_filters(image):
    cv2 = _load_opencv()
    rgb = np.asarray(resize_for_analysis(image))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(rgb, (15, 15), 0)
    sharpened = cv2.addWeighted(rgb, 1.5, blurred, -0.5, 0)
    threshold = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    contours_image = rgb.copy()
    contours, _ = cv2.findContours(
        cv2.Canny(gray, 50, 150), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(contours_image, contours, -1, (240, 180, 41), 1)
    return {"Grayscale": gray, "Blurred": blurred, "Sharpened": sharpened,
            "Threshold": threshold, "Contours": contours_image}
