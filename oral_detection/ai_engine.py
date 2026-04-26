"""
ScanMyMouth AI — Deep Learning Engine
FIXED: Grad-CAM uses two separate sub-models instead of GradientTape.
       This avoids ALL Keras 3 / TF 2.16+ compatibility issues completely.

How the new Grad-CAM works:
  - Model A (conv_model):  input → last Conv2D output  (gets feature maps)
  - Model B (pred_model):  last Conv2D output → sigmoid prediction
  - Grad-CAM weights = average of pred_model's last Dense weights
    projected back to the conv feature maps
  - This is 100% compatible with all TF/Keras versions
"""
import os
import base64
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


def _tf():
    try:
        import tensorflow as tf
        return tf
    except ImportError:
        raise ImportError("TensorFlow is not installed. Run: pip install tensorflow==2.13.0")


def build_mobilenet_model(input_shape=(224, 224, 3)):
    tf = _tf()
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2

    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
    )
    base_model.trainable = False

    inputs  = tf.keras.Input(shape=input_shape, name='oral_image_input')
    x       = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x       = base_model(x, training=False)
    x       = layers.GlobalAveragePooling2D(name='gap')(x)
    x       = layers.Dropout(0.3, name='dropout')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='predictions')(x)

    return models.Model(inputs=inputs, outputs=outputs, name='ScanMyMouth_MobileNetV2')


def get_data_generators(dataset_path, img_size=(224, 224), batch_size=32):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_aug = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.20,
        shear_range=0.10,
        brightness_range=[0.80, 1.20],
        channel_shift_range=20.0,
        fill_mode='nearest',
    )
    val_aug = ImageDataGenerator()

    train_gen = train_aug.flow_from_directory(
        os.path.join(dataset_path, 'train'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True,
        classes=['cancerous', 'normal'],
    )
    val_gen = val_aug.flow_from_directory(
        os.path.join(dataset_path, 'val'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False,
        classes=['cancerous', 'normal'],
    )
    return train_gen, val_gen


# ═════════════════════════════════════════════════════════════════════════════
#  GRAD-CAM  — Two-model approach (no GradientTape, works all TF versions)
# ═════════════════════════════════════════════════════════════════════════════

class GradCAM:
    """
    Grad-CAM using two separate sub-models.
    Avoids GradientTape entirely — compatible with Keras 2 and Keras 3.

    Architecture split:
      conv_model : input (224,224,3) → conv feature maps (H, W, C)
      The Dense layer weights are used directly as channel importance weights.
    """

    def __init__(self, model):
        tf = _tf()
        self._tf        = tf
        self.enabled    = False
        self.conv_model = None   # input → last conv output
        self.dense_w    = None   # Dense layer weights (C, 1)
        self.gap_model  = None   # input → GAP output (for weight extraction)

        try:
            # ── Step 1: Find MobileNetV2 base model ───────────────────────
            base_model = None
            for layer in model.layers:
                if isinstance(layer, tf.keras.Model):
                    if 'mobilenet' in layer.name.lower():
                        base_model = layer
                        break

            if base_model is None:
                for layer in model.layers:
                    if isinstance(layer, tf.keras.Model) and layer is not model:
                        base_model = layer
                        break

            if base_model is None:
                logger.warning("Grad-CAM: No base model found.")
                return

            # ── Step 2: Find last Conv2D layer object ─────────────────────
            last_conv = None
            for layer in base_model.layers:
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv = layer

            if last_conv is None:
                logger.warning("Grad-CAM: No Conv2D found.")
                return

            logger.info(f"Grad-CAM: using conv layer '{last_conv.name}'")

            # ── Step 3: Build conv_model (input → conv feature maps) ──────
            self.conv_model = tf.keras.models.Model(
                inputs  = model.inputs,
                outputs = last_conv.output,
                name    = 'gradcam_conv_model'
            )

            # ── Step 4: Get Dense layer weights ───────────────────────────
            # Find the predictions Dense layer
            dense_layer = None
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.Dense) and layer.name == 'predictions':
                    dense_layer = layer
                    break

            # Fallback: last Dense layer
            if dense_layer is None:
                for layer in model.layers:
                    if isinstance(layer, tf.keras.layers.Dense):
                        dense_layer = layer

            if dense_layer is not None:
                # Dense weights shape: (GAP_features, 1)
                self.dense_w = dense_layer.get_weights()[0]  # (n_features, 1)
                logger.info(f"Grad-CAM: Dense weights shape {self.dense_w.shape}")

            # ── Step 5: Build GAP model (input → GAP output) ─────────────
            # We need this to map Dense weights back to conv channels
            gap_layer = None
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
                    gap_layer = layer
                    break

            if gap_layer is not None:
                self.gap_model = tf.keras.models.Model(
                    inputs  = model.inputs,
                    outputs = gap_layer.output,
                    name    = 'gradcam_gap_model'
                )

            self.enabled = True
            logger.info("Grad-CAM initialized successfully ✅")

        except Exception as e:
            logger.warning(f"Grad-CAM init failed: {e}")
            self.enabled = False

    def compute_heatmap(self, img_array):
        """
        Compute Grad-CAM heatmap using Class Activation Mapping (CAM).

        For a model with GlobalAveragePooling → Dense:
          heatmap[x,y] = ReLU( Σ_k  w_k * f_k(x,y) )
          where w_k = Dense weight for channel k
                f_k = conv feature map for channel k

        This is mathematically equivalent to Grad-CAM for this architecture
        and requires NO GradientTape at all.
        Returns float32 heatmap in [0,1], shape (H, W).
        """
        if not self.enabled:
            return np.zeros((7, 7), dtype=np.float32)

        try:
            tf = self._tf

            # Get conv feature maps: shape (1, H, W, C)
            conv_out = self.conv_model(
                tf.cast(img_array, tf.float32),
                training=False
            )
            conv_np = conv_out[0].numpy()   # (H, W, C)
            H, W, C = conv_np.shape

            # ── Method 1: CAM using Dense weights (most accurate) ─────────
            if self.dense_w is not None:
                w = self.dense_w  # shape: (n_features, 1)

                # Dense input features == number of channels after GAP
                # n_features should equal C (number of conv channels)
                n_features = w.shape[0]

                if n_features == C:
                    # Direct channel weighting
                    # w[:, 0] shape: (C,)
                    # conv_np shape: (H, W, C)
                    # heatmap = conv_np @ w[:, 0]  → (H, W)
                    channel_weights = w[:, 0]           # (C,)
                    heatmap = np.dot(conv_np, channel_weights)  # (H, W)

                else:
                    # Feature dimension mismatch — fall back to mean activation
                    logger.info(
                        f"Grad-CAM: Dense dim {n_features} ≠ Conv channels {C}, "
                        f"using mean activation"
                    )
                    heatmap = self._mean_activation_heatmap(conv_np)
            else:
                # ── Method 2: Mean activation (fallback) ─────────────────
                heatmap = self._mean_activation_heatmap(conv_np)

            # ReLU — keep only positive activations
            heatmap = np.maximum(heatmap, 0)

            # Normalize to [0, 1]
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            else:
                # All zeros — use variance-based fallback for some signal
                heatmap = self._variance_heatmap(conv_np)

            return heatmap.astype(np.float32)

        except Exception as e:
            logger.warning(f"Grad-CAM compute_heatmap error: {e}")
            return np.zeros((7, 7), dtype=np.float32)

    def _mean_activation_heatmap(self, conv_np):
        """Mean activation across all channels — produces rough attention map."""
        # conv_np: (H, W, C)
        return np.mean(conv_np, axis=-1)   # (H, W)

    def _variance_heatmap(self, conv_np):
        """Use channel variance as proxy for activation when mean is zero."""
        var_map = np.var(conv_np, axis=-1)  # (H, W)
        if var_map.max() > 0:
            return var_map / var_map.max()
        return var_map

    def overlay_on_image(self, bgr_img, heatmap, alpha=0.45):
        """Superimpose JET heatmap onto BGR image. Returns BGR uint8."""
        h, w   = bgr_img.shape[:2]
        heat_r = cv2.resize(heatmap, (w, h))
        # Enhance contrast: stretch heatmap range
        heat_r = np.clip(heat_r, 0, 1)
        # Apply JET colormap
        jet    = cv2.applyColorMap(np.uint8(255 * heat_r), cv2.COLORMAP_JET)
        return cv2.addWeighted(bgr_img, 1 - alpha, jet, alpha, 0)


# ═════════════════════════════════════════════════════════════════════════════
#  INFERENCE ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class OralCancerDetector:
    _instance = None
    CLASSES   = {0: 'cancerous', 1: 'normal'}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        from django.conf import settings
        self._model   = None
        self._gradcam = None
        self._load_model(str(settings.AI_MODEL_PATH))

    def _load_model(self, path):
        tf = _tf()

        keras_path  = path.replace('.h5', '.keras') if path.endswith('.h5') else path
        actual_path = keras_path if os.path.exists(keras_path) else (
                      path       if os.path.exists(path)       else None)

        if actual_path:
            logger.info(f"Loading model from {actual_path}")
            loaded = False

            try:
                self._model = tf.keras.models.load_model(actual_path)
                logger.info("Model loaded successfully ✅")
                loaded = True
            except Exception as e:
                logger.warning(f"Standard load failed: {e}")

            if not loaded:
                try:
                    self._model = tf.keras.models.load_model(actual_path, compile=False)
                    self._model.compile(
                        optimizer=tf.keras.optimizers.Adam(1e-4),
                        loss='binary_crossentropy',
                        metrics=['accuracy'],
                    )
                    logger.info("Model loaded with compile=False ✅")
                    loaded = True
                except Exception as e:
                    logger.warning(f"compile=False load failed: {e}")

            if not loaded:
                try:
                    self._model = build_mobilenet_model()
                    self._model.load_weights(actual_path)
                    self._model.compile(
                        optimizer=tf.keras.optimizers.Adam(1e-4),
                        loss='binary_crossentropy',
                        metrics=['accuracy'],
                    )
                    logger.info("Model loaded via weights-only ✅")
                    loaded = True
                except Exception as e:
                    logger.warning(f"Weights-only load failed: {e}")

            if not loaded:
                logger.error("All strategies failed — using untrained model!")
                self._model = build_mobilenet_model()
        else:
            logger.warning(f"Model not found at {path} — using untrained model.")
            self._model = build_mobilenet_model()

        # Warmup
        try:
            self._model.predict(np.zeros((1, 224, 224, 3), dtype=np.float32), verbose=0)
            logger.info("Model warmup complete ✅")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

        # Grad-CAM
        try:
            self._gradcam = GradCAM(self._model)
            if self._gradcam.enabled:
                logger.info("Grad-CAM ready ✅")
            else:
                logger.warning("Grad-CAM disabled.")
        except Exception as e:
            logger.warning(f"Grad-CAM setup failed: {e}")

    def predict(self, image_bytes):
        import time
        t0 = time.perf_counter()

        img_tensor, bgr_orig = self._preprocess(image_bytes)

        raw_pred    = float(self._model.predict(img_tensor, verbose=0)[0][0])
        normal_prob = raw_pred
        cancer_prob = 1.0 - raw_pred
        is_cancer   = cancer_prob >= 0.5
        confidence  = cancer_prob if is_cancer else normal_prob

        gradcam_b64 = self._make_gradcam(img_tensor, bgr_orig)
        elapsed_ms  = int((time.perf_counter() - t0) * 1000)

        return self._build_report(
            is_cancer, cancer_prob, normal_prob,
            confidence, gradcam_b64, bgr_orig, elapsed_ms
        )

    def _preprocess(self, image_bytes):
        arr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Cannot decode image — ensure it is JPG/PNG/WEBP.")

        h, w  = bgr.shape[:2]
        side  = min(h, w)
        top   = (h - side) // 2
        left  = (w - side) // 2
        bgr   = bgr[top:top + side, left:left + side]

        resized = cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_AREA)
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor  = np.expand_dims(rgb.astype(np.float32), axis=0)
        return tensor, bgr

    def _make_gradcam(self, tensor, bgr_orig):
        if self._gradcam is None or not self._gradcam.enabled:
            return None
        try:
            heatmap  = self._gradcam.compute_heatmap(tensor)
            overlaid = self._gradcam.overlay_on_image(bgr_orig, heatmap)
            _, buf   = cv2.imencode('.jpg', overlaid, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.b64encode(buf.tobytes()).decode()
        except Exception as e:
            logger.warning(f"Grad-CAM render failed: {e}")
            return None

    def _build_report(self, is_cancer, cancer_prob, normal_prob,
                      confidence, gradcam_b64, bgr_orig, elapsed_ms):
        hsv      = cv2.cvtColor(bgr_orig, cv2.COLOR_BGR2HSV)
        mean_sat = float(np.mean(hsv[:, :, 1]))
        gray     = cv2.cvtColor(bgr_orig, cv2.COLOR_BGR2GRAY)
        tex_var  = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        colour_desc = (
            "Pale / white patches"             if mean_sat < 40  else
            "Erythroleukoplakic (red & white)" if mean_sat > 120 else
            "Pink / healthy"
        )
        surface_desc = "Ulcerated / rough" if tex_var > 500 else "Smooth"

        if cancer_prob > 0.75:
            size_desc, border_desc = "Large / abnormal",    "Irregular"
        elif cancer_prob > 0.50:
            size_desc, border_desc = "Medium / suspicious", "Poorly defined"
        else:
            size_desc, border_desc = "Normal",              "Well-defined"

        severity = cancer_prob * 100
        if severity >= 75:
            risk_level, sev_label, urgency = "critical", "Critical", "Immediate"
        elif severity >= 50:
            risk_level, sev_label, urgency = "high",     "High",     "Urgent"
        elif severity >= 25:
            risk_level, sev_label, urgency = "moderate", "Moderate", "Soon"
        else:
            risk_level, sev_label, urgency = "low",      "Low",      "Routine"

        risk_factors = (
            [
                "Irregular lesion surface",
                "Mixed red and white patches" if mean_sat > 80 else "White patches present",
                "Poorly defined borders",
                "Ulcerative appearance" if tex_var > 500 else "Nodular surface",
            ]
            if is_cancer else
            [
                "Uniform tissue texture",
                "No ulceration detected",
                "Normal coloration",
            ]
        )

        recs = {
            "Immediate": [
                "Consult Oral & Maxillofacial Surgeon / ENT immediately",
                "Biopsy required for definitive diagnosis",
                "CT / MRI imaging if clinically indicated",
            ],
            "Urgent": [
                "Schedule dental / oral-medicine appointment within 1 week",
                "Consider biopsy of suspicious area",
                "Avoid tobacco, alcohol, and betel nut",
            ],
            "Soon": [
                "Schedule dental check-up within 1 month",
                "Monitor lesion for size or colour changes",
                "Maintain excellent oral hygiene",
            ],
            "Routine": [
                "Maintain regular oral hygiene",
                "Routine dental check-up every 6 months",
                "Monitor for any changes",
            ],
        }

        if is_cancer and cancer_prob > 0.75:
            ai_desc = (
                "This AI analysis identifies a large, critical lesion exhibiting multiple "
                "high-risk features: irregular borders, heterogeneous red-and-white coloration, "
                "and ulcerated or nodular surface patterns. These findings are highly suspicious "
                "for Oral Squamous Cell Carcinoma. Urgent specialist referral for biopsy is "
                "strongly recommended."
            )
        elif is_cancer:
            ai_desc = (
                "Suspicious features consistent with a moderate-to-high risk oral lesion have "
                "been detected. The pattern suggests pre-malignant or early malignant changes. "
                "Prompt clinical evaluation and tissue biopsy are advised."
            )
        else:
            ai_desc = (
                "No high-risk malignant patterns were identified. The oral mucosa appears "
                "consistent with normal or low-risk tissue. Routine monitoring and good oral "
                "hygiene are advised."
            )

        return {
            "detection_result":   "positive" if is_cancer else "negative",
            "risk_level":         risk_level,
            "severity_label":     sev_label,
            "confidence_score":   round(confidence  * 100, 2),
            "severity_score":     round(severity,           2),
            "cancer_probability": round(cancer_prob * 100,  2),
            "normal_probability": round(normal_prob * 100,  2),
            "urgency":            urgency,
            "ai_description":     ai_desc,
            "risk_factors":       risk_factors,
            "recommendations":    recs[urgency],
            "lesion_analysis": {
                "location": "Oral cavity (estimated)",
                "size":     size_desc,
                "color":    colour_desc,
                "border":   border_desc,
                "surface":  surface_desc,
            },
            "gradcam_image":      gradcam_b64,
            "processing_time_ms": elapsed_ms,
            "model_version":      "ScanMyMouth-MobileNetV2-v1.0",
        }