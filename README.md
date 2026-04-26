# 🦷 ScanMyMouth AI
### Intelligent Diagnosis for Oral Health Risks
**Final Year Project — Deep Learning · TensorFlow · OpenCV · Django**

---

## 📁 Project Structure

```
scanmymouth_ai/
├── requirements.txt
├── prepare_dataset.py          ← Run ONCE after downloading dataset
│
└── backend/
    ├── manage.py
    ├── scanmymouth/            ← Django project config
    │   ├── settings.py
    │   ├── urls.py
    │   └── wsgi.py
    ├── oral_detection/         ← Main Django app
    │   ├── ai_engine.py        ← CNN + Grad-CAM (TF + OpenCV)
    │   ├── models.py           ← Database models
    │   ├── views.py            ← REST API endpoints
    │   ├── urls.py
    │   ├── serializers.py
    │   ├── admin.py
    │   ├── migrations/
    │   └── management/
    │       └── commands/
    │           └── train_model.py   ← Training CLI
    ├── templates/
    │   ├── index.html          ← Upload page
    │   └── results.html        ← Results page
    └── media/                  ← Uploaded images & Grad-CAM outputs
```

---

## 🚀 Quick Start (Step-by-Step)

### Step 1 — Clone & Environment

```bash
# Navigate to project
cd scanmymouth_ai

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

### Step 2 — Get the Dataset

You have **three free options**. Use any one:

---

#### ✅ Option A — Kaggle: Oral Cancer (Lips and Tongue) [RECOMMENDED]

1. Create a free Kaggle account at https://www.kaggle.com
2. Go to: https://www.kaggle.com/datasets/shubhamgoel27/dermnet
   OR search: **"oral cancer image dataset kaggle"**
3. Best dataset: https://www.kaggle.com/datasets/ashenafifasilkebede/dataset
   - ~3,000+ images, 2 classes: `cancerous` and `non_cancerous`
4. Download and unzip to `/path/to/raw_dataset/`

Expected layout after unzip:
```
raw_dataset/
    cancerous/
        img001.jpg
        img002.jpg
        ...
    non_cancerous/
        img001.jpg
        ...
```

Prepare it:
```bash
cd scanmymouth_ai
python prepare_dataset.py \
    --source /path/to/raw_dataset \
    --dest   /path/to/dataset \
    --cancer-dir cancerous \
    --normal-dir non_cancerous \
    --augment
```

---

#### ✅ Option B — Kaggle: Oral Cancer Detection Dataset

URL: https://www.kaggle.com/datasets/zaidpy/oral-cancer-dataset

Layout:
```
raw_dataset/
    Cancer/
    Normal/
```

Prepare:
```bash
python prepare_dataset.py \
    --source /path/to/raw_dataset \
    --dest   /path/to/dataset \
    --cancer-dir Cancer \
    --normal-dir Normal \
    --augment
```

---

#### ✅ Option C — GitHub ORCA Dataset (Academic)

URL: https://github.com/srikarym/oral-cancer-detection

Follow their README to download, then run prepare_dataset.py.

---

#### After preparation, your dataset must look like this:

```
/path/to/dataset/
    train/
        cancerous/       ← ~70% of cancerous images
        normal/          ← ~70% of normal images
    val/
        cancerous/       ← ~15%
        normal/          ← ~15%
    test/
        cancerous/       ← ~10%
        normal/          ← ~10%
```

---

### Step 3 — Database Setup

```bash
cd backend
python manage.py migrate
python manage.py createsuperuser   # optional: for admin panel
```

---

### Step 4 — Train the Model

#### Option A: EfficientNetB3 Transfer Learning (RECOMMENDED — Higher Accuracy)

```bash
python manage.py train_model \
    --dataset /path/to/dataset \
    --epochs  50 \
    --batch   32
```

#### Option B: Custom CNN from Scratch

```bash
python manage.py train_model \
    --dataset   /path/to/dataset \
    --epochs    100 \
    --batch     32 \
    --no-transfer
```

Training will:
- Show live accuracy, AUC, precision, recall per epoch
- Save the best model automatically to `oral_detection/models_dir/oral_cancer_model.h5`
- Use early stopping (patience=12) and ReduceLROnPlateau
- Write `training_log.csv` with full epoch history
- Write TensorBoard logs to `./logs/`

#### Monitor with TensorBoard (optional):
```bash
tensorboard --logdir ./logs
# Open: http://localhost:6006
```

Expected results after training on a good dataset:
```
Val Accuracy : ~88–94%
Val AUC      : ~0.92–0.97
```

---

### Step 5 — Run the Server

```bash
cd backend
python manage.py runserver
```

Open: **http://127.0.0.1:8000/**

---

## 🔌 REST API Reference

All endpoints are prefixed with `/api/`

| Method | Endpoint          | Description                        |
|--------|-------------------|------------------------------------|
| POST   | `/api/analyze/`   | Upload image, get AI analysis      |
| GET    | `/api/history/`   | Last 50 analysis records           |
| GET    | `/api/history/<uuid>/` | Single record by ID          |
| GET    | `/api/health/`    | Model health check                 |
| GET    | `/api/stats/`     | Aggregate statistics               |

### POST /api/analyze/ — Example

```bash
curl -X POST http://127.0.0.1:8000/api/analyze/ \
     -F "image=@/path/to/oral_image.jpg"
```

### Response Schema

```json
{
  "analysis_id":        "uuid-string",
  "detection_result":   "positive" | "negative",
  "risk_level":         "low" | "moderate" | "high" | "critical",
  "severity_label":     "Low" | "Moderate" | "High" | "Critical",
  "confidence_score":   87.5,
  "severity_score":     72.3,
  "cancer_probability": 72.3,
  "normal_probability": 27.7,
  "urgency":            "Immediate" | "Urgent" | "Soon" | "Routine",
  "ai_description":     "Full AI narrative...",
  "risk_factors":       ["Irregular lesion surface", "..."],
  "recommendations":    ["Consult specialist", "..."],
  "lesion_analysis": {
    "location": "Oral cavity (estimated)",
    "size":     "Large / abnormal",
    "color":    "Erythroleukoplakic (red & white)",
    "border":   "Irregular",
    "surface":  "Ulcerated / rough"
  },
  "gradcam_image":      "base64-encoded-jpeg-string",
  "gradcam_image_url":  "http://...",
  "original_image_url": "http://...",
  "processing_time_ms": 1842,
  "model_version":      "ScanMyMouth-CNN-v1.0",
  "created_at":         "2026-04-05T10:23:41.000Z"
}
```

---

## 🧠 Model Architecture

### Custom CNN (4 Blocks)

```
Input (224×224×3)
  → ConvBlock1: Conv2D(32) × 2 → BN → MaxPool → Dropout(0.25)
  → ConvBlock2: Conv2D(64) × 2 → BN → MaxPool → Dropout(0.25)
  → ConvBlock3: Conv2D(128)× 3 → BN → MaxPool → Dropout(0.30)
  → ConvBlock4: Conv2D(256)× 3 → BN → MaxPool → Dropout(0.40)  ← Grad-CAM target: 'last_conv'
  → GlobalAveragePooling2D
  → Dense(512) → BN → Dropout(0.50)
  → Dense(256) → Dropout(0.30)
  → Dense(2, softmax)  ← [cancerous_prob, normal_prob]
```

### Transfer Learning (EfficientNetB3)

```
Input (224×224×3)
  → EfficientNetB3 (ImageNet weights, last 30 layers unfrozen)
  → GlobalAveragePooling2D
  → Dense(512, relu) → Dropout(0.50)
  → Dense(256, relu) → Dropout(0.30)
  → Dense(2, softmax)
```

---

## 🔥 Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM) generates a visual explanation
of the CNN's decision by computing gradients of the predicted class score with respect
to the last convolutional layer activations.

Key properties:
- Target layer: `last_conv` (final Conv2D in Block 4)
- Heatmap uses JET colormap: red = high activation, blue = low
- Overlay alpha = 0.45 (45% heatmap, 55% original image)

---

## ⚙️ Configuration

Edit `backend/scanmymouth/settings.py`:

```python
AI_MODEL_PATH        = BASE_DIR / 'oral_detection' / 'models_dir' / 'oral_cancer_model.h5'
AI_INPUT_SIZE        = (224, 224)
AI_CONFIDENCE_THRESHOLD = 0.5      # decision boundary
```

---

## 🗃️ Admin Panel

```bash
python manage.py createsuperuser
# Open: http://127.0.0.1:8000/admin/
```

View all analysis records, filter by result/risk level, search by description.

---

## 📦 Dependencies

| Package                   | Purpose                          |
|---------------------------|----------------------------------|
| Django 4.2                | Web framework + ORM              |
| djangorestframework 3.14  | REST API                         |
| django-cors-headers       | CORS for frontend                |
| TensorFlow 2.13           | CNN training & inference         |
| OpenCV (headless)         | Image preprocessing & Grad-CAM   |
| NumPy                     | Array operations                 |
| Pillow                    | Image field support              |
| scikit-learn              | Class weight computation         |

---

## 📋 Dataset Tips

- Minimum recommended: **500 images per class** (1000 total)
- Best performance: **2000+ images per class**
- Use `--augment` flag in prepare_dataset.py if one class has fewer images
- Images should be clear, well-lit oral cavity photographs
- Avoid blurry, low-resolution, or heavily compressed images
- JPEG quality ≥ 70 recommended

---

## ⚠️ Medical Disclaimer

This tool is for **educational and screening purposes only**. It is **not a substitute
for professional medical diagnosis**. All results must be reviewed by a qualified oral
health specialist. Do not make clinical decisions based solely on this tool.

---

*© 2026 ScanMyMouth AI — Final Year Project*
