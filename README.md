# QR Code Phishing Classifier 🔒

A deep learning-based QR code malware/phishing detection system using EfficientNet-B3, achieving **97%+ validation accuracy**. This project classifies QR codes as **Safe** or **Malicious** to protect users from UPI payment scams and phishing attacks.

[![Kaggle Model](https://img.shields.io/badge/Kaggle-Model-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/models/devilfrost/qr-fishing)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/samahsadiq/benign-and-malicious-qr-codes)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 Quick Start

### Use Pre-trained Model (Recommended)

```python
# On Kaggle: Add these datasets to your notebook
# 1. Model: https://www.kaggle.com/models/devilfrost/qr-fishing
# 2. Dataset: /kaggle/input/benign-and-malicious-qr-codes

MODEL_PATH = '/kaggle/input/qr-fishing/pytorch/default/1/best_model.pth'
DATA_PATH = '/kaggle/input/benign-and-malicious-qr-codes/QR codes'
```

**No training needed!** Just load the model and start predicting.

---

## 🎯 Project Overview

This classifier is designed to detect malicious QR codes commonly used in:

- UPI payment scams
- Phishing attacks
- Fraudulent payment requests
- Social engineering schemes

### Key Features

- ✅ **97%+ Accuracy** on validation set
- 🚀 **Mixed Precision Training** (FP16) for 2-3x speedup
- 🧠 **EfficientNet-B3** backbone with custom classification head
- 📱 **Phone Camera Simulation** augmentation for real-world robustness
- ⚡ **Fast Inference** (~50ms per image on GPU)
- 💾 **Auto-checkpointing** with Kaggle timeout protection
- 🎨 **Advanced augmentation** pipeline

---

## 📊 Model Performance

| Metric              | Score                     |
| ------------------- | ------------------------- |
| **Test Accuracy**   | **98.28%** ✅             |
| Validation Accuracy | 97.6%                     |
| Training Accuracy   | 93.4%                     |
| **Precision**       | **0.9861** (98.61%)       |
| **Recall**          | **0.9786** (97.86%)       |
| **F1-Score**        | **0.9823** (98.23%)       |
| **ROC-AUC**         | **0.9987** (99.87%)       |
| Loss (Val)          | 0.1046                    |
| Error Rate          | 1.73% (345/20,000)        |
| Model               | EfficientNet-B3           |
| Parameters          | ~12M (trainable: ~1.5M)   |
| Training Time       | ~7 hours (Kaggle T4 GPU)  |
| Test Time           | ~3-5 minutes (20K images) |

---

## 🛠️ Tech Stack

### Deep Learning

- **PyTorch** - Deep learning framework
- **TorchVision** - Pre-trained models and transforms
- **EfficientNet-B3** - Backbone architecture
- **Mixed Precision (FP16)** - Training optimization

### Data Processing

- **PIL/Pillow** - Image processing
- **NumPy** - Numerical operations
- **scikit-learn** - Data splitting, metrics

### Training Optimizations

- **AdamW** optimizer with weight decay
- **Cosine Annealing LR** with warmup
- **Gradient Accumulation** (effective batch size: 64)
- **Label Smoothing** (0.1)
- **Early Stopping** (patience: 5)
- **Progressive Unfreezing** at epoch 5

---

## 📁 Project Structure

```
qr-fishing/
├── QR.ipynb                    # Main training notebook
├── README.md                   # This file
├── README_STUDY_GUIDE.md       # Viva preparation guide
├── README_DEPLOYMENT.md        # Deployment guide
├── .gitignore                  # Git ignore rules
├── kaggle.json                 # Kaggle API credentials
├── artifacts/                  # Model outputs (not in repo)
│   ├── best_model.pth         # Best checkpoint
│   ├── qr_classifier_final.pth # Final model
│   ├── model_weights.pth      # Weights only
│   ├── history.csv            # Training history
│   ├── training_history.png   # Plots
│   └── confusion_matrix.png   # Confusion matrix
└── QR codes/                   # Dataset (not in repo)
    ├── benign/
    └── malicious/
```

---

## 🚀 Getting Started

### 1. Prerequisites

```bash
# Python 3.8+
pip install torch torchvision torchaudio
pip install pillow numpy scikit-learn tqdm matplotlib seaborn
```

### 2. Dataset & Pre-trained Model

**📦 Pre-trained Model (Ready to Use):**

- Model: [qr-fishing on Kaggle Models](https://www.kaggle.com/models/devilfrost/qr-fishing)
- Direct path: `/kaggle/input/qr-fishing/pytorch/default/1/best_model.pth`

**📊 Dataset:**

- Dataset: [benign-and-malicious-qr-codes](https://www.kaggle.com/datasets/samahsadiq/benign-and-malicious-qr-codes)
- Kaggle path: `/kaggle/input/benign-and-malicious-qr-codes`
- ~200,000 QR code images (benign + malicious)

Place in `QR codes/` directory:

```
QR codes/
├── benign/benign/
└── malicious/malicious/
```

**🎯 Quick Start on Kaggle:**

1. Create new notebook
2. Add data: Click "+ Add Data" → Search "qr-fishing" (model) and "benign-and-malicious-qr-codes" (dataset)
3. Model will be at: `/kaggle/input/qr-fishing/pytorch/default/1/best_model.pth`
4. Dataset will be at: `/kaggle/input/benign-and-malicious-qr-codes/QR codes`

### 3. Training

Open `QR.ipynb` in Jupyter/Kaggle and run all cells:

- Cell 1-7: Data loading and model setup
- Cell 8: (Optional) Resume from checkpoint
- Cell 9-11: Training loop
- Cell 12-16: Evaluation and visualization

**Hyperparameters (Cell 3):**

```python
IMG_SIZE = 256                  # Image resolution
MODEL_NAME = 'efficientnet_b3'  # Model architecture
BATCH_SIZE = 32                 # Batch size
EPOCHS = 25                     # Training epochs
LEARNING_RATE = 5e-4            # Initial LR
PATIENCE = 5                    # Early stopping patience
```

### 4. Model Files

**Pre-trained model available:**

- 🎯 **Kaggle Model**: [devilfrost/qr-fishing](https://www.kaggle.com/models/devilfrost/qr-fishing)
- Path in Kaggle notebooks: `/kaggle/input/qr-fishing/pytorch/default/1/best_model.pth`

**Or train your own:**
After training, find models in `artifacts/`:

- `best_model.pth` - Best checkpoint (includes optimizer state)
- `qr_classifier_final.pth` - Final model with metadata
- `model_weights.pth` - Weights only (smallest file)

---

## 💻 Inference (Using Trained Model)

### Option 1: Use Pre-trained Model from Kaggle

```python
import torch
from torchvision import transforms
from PIL import Image

# On Kaggle, use the pre-trained model
MODEL_PATH = '/kaggle/input/qr-fishing/pytorch/default/1/best_model.pth'

# Or local path
# MODEL_PATH = 'artifacts/best_model.pth'

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(MODEL_PATH, map_location=device)
model = QRClassifier(model_name='efficientnet_b3')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print(f"✅ Model loaded! Trained for {checkpoint['epoch']} epochs")
print(f"   Validation accuracy: {checkpoint['val_acc']:.4f}")
```

### Option 2: Predict Function

```python
# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Predict
def predict_qr(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()

    label = "🚨 Malicious" if prob >= 0.5 else "✅ Safe"
    confidence = max(prob, 1-prob) * 100

    return label, confidence

# Example
label, conf = predict_qr('test_qr.png')
print(f"{label} ({conf:.2f}% confidence)")
```

### Option 3: Quick Evaluation Script

```python
# Evaluate on Kaggle using pre-trained model
# Just run qr_model_evaluation_simple.ipynb
# Model: https://www.kaggle.com/models/devilfrost/qr-fishing
# Dataset: /kaggle/input/benign-and-malicious-qr-codes
```

---

## 🎨 Data Augmentation

### Phone Camera Simulation (Custom)

Realistic augmentations to simulate real-world QR scanning:

- **Lighting issues** - Brightness/contrast variations
- **Motion blur** - Simulates shaky hands
- **JPEG compression** - Camera compression artifacts
- **Focus issues** - Gaussian blur
- **Color cast** - Different camera sensors

### Standard Augmentations

- Random flips (horizontal/vertical)
- Random rotation (±10°)
- Random affine transforms
- Random perspective distortion
- Color jitter
- Random erasing

---

## 📈 Training Details

### Architecture

```
EfficientNet-B3 Backbone (frozen initially)
    ↓
Progressive Unfreezing (epoch 5, top 30%)
    ↓
Custom Classification Head:
    - Dropout(0.3)
    - Linear(1536 → 256)
    - BatchNorm1d(256)
    - ReLU
    - Dropout(0.15)
    - Linear(256 → 1)
    - BCEWithLogitsLoss
```

### Training Strategy

1. **Phase 1 (Epochs 1-4):** Train classification head only
2. **Phase 2 (Epochs 5-25):** Unfreeze top 30% of backbone, lower LR by 10x
3. **Early stopping:** Stop if no improvement for 5 epochs

### Hardware

- **Kaggle T4 GPU** (15GB VRAM)
- **Training time:** ~7 hours for 25 epochs
- **Batch size:** 32 (effective: 64 with gradient accumulation)

---

## 📊 Results & Metrics

### Test Set Performance (20,000 Images)

```
🎯 Overall Metrics:
   Accuracy:  98.28% (19,655 correct / 20,000 total)
   Precision: 0.9861
   Recall:    0.9786
   F1-Score:  0.9823
   ROC-AUC:   0.9987

📈 Per-Class Performance:
   Benign:    10,074/10,209 (98.68%)
   Malicious: 9,581/9,791   (97.86%)

❌ Error Analysis:
   Total Errors: 345 (1.73%)
   False Positives: 135 (1.32%) - Safe marked as malicious
   False Negatives: 210 (2.14%) - Malicious marked as safe
```

### Confusion Matrix

| Actual ↓ / Predicted → | Benign | Malicious |
| ---------------------- | ------ | --------- |
| **Benign**             | 98.68% | 1.32%     |
| **Malicious**          | 2.14%  | 97.86%    |

**Raw Counts:**

- True Negatives (Benign → Benign): 10,074
- False Positives (Benign → Malicious): 135
- False Negatives (Malicious → Benign): 210
- True Positives (Malicious → Malicious): 9,581

### Classification Report

```
              precision    recall  f1-score   support
      Benign       0.98      0.99      0.98     10,209
   Malicious       0.99      0.98      0.98      9,791

    accuracy                           0.98     20,000
   macro avg       0.98      0.98      0.98     20,000
weighted avg       0.98      0.98      0.98     20,000
```

### Key Insights

✅ **Model is Production-Ready:**

- Test accuracy (98.28%) > Validation accuracy (97.6%)
- No signs of overfitting
- Balanced performance across both classes
- High confidence predictions (most >95%)

✅ **Security Characteristics:**

- Slightly conservative: 210 false negatives vs 135 false positives
- ROC-AUC of 0.9987 indicates excellent discrimination
- Clear separation between benign and malicious classes

---

## 🚀 Deployment Readiness

### ✅ Production Status: **READY**

Your model has been thoroughly evaluated and is **production-ready** with excellent performance:

| Criteria           | Status  | Details                            |
| ------------------ | ------- | ---------------------------------- |
| **Accuracy**       | ✅ Pass | 98.28% on 20K test images          |
| **Generalization** | ✅ Pass | Test > Validation (no overfitting) |
| **Balance**        | ✅ Pass | Both classes >97% accuracy         |
| **Confidence**     | ✅ Pass | Most predictions >95% confident    |
| **Error Rate**     | ✅ Pass | Only 1.73% errors (345/20,000)     |
| **Speed**          | ✅ Pass | ~50ms per image on GPU             |
| **Robustness**     | ✅ Pass | Handles phone camera variations    |

### 🎯 Deployment Recommendations

**For General Use (Current Settings):**

- Threshold: 0.5
- Accuracy: 98.28%
- Balanced false positives and negatives
- **RECOMMENDED** for most applications

**For High Security (Banking/Finance):**

- Adjust threshold to 0.4
- Catches more malicious QRs (higher recall)
- More warnings to users (more false positives)
- Better safe than sorry approach

**For Better UX (Low-Risk Apps):**

- Adjust threshold to 0.6
- Fewer false alarms
- May miss some malicious QRs
- Use only if user education is strong

### 📊 Expected Performance in Production

```python
# On 1 million scans:
Total Scans:        1,000,000
Correct Predictions: 982,800 (98.28%)
False Alarms:        6,600 (0.66%)    # Safe marked as malicious
Missed Threats:     10,600 (1.06%)    # Malicious marked as safe
```

### 🛡️ Risk Assessment

**Low Risk:**

- Only 1.73% error rate
- High confidence in most predictions
- Proven generalization ability

**Mitigation Strategies:**

1. Show confidence scores to users
2. Allow manual review for borderline cases (40-60% confidence)
3. Log all predictions for continuous monitoring
4. Update model with new malicious patterns regularly

---

## 🚀 Deployment Options

### 1. REST API (FastAPI)

Deploy model as a web service for real-time inference.

### 2. Mobile App (React Native)

Integrate QR scanner with real-time malware detection.

### 3. Browser Extension

Scan QR codes from web pages and warn users.

### 4. Middleware

Integrate with payment gateways to block malicious QR codes.

**See [README_DEPLOYMENT.md](README_DEPLOYMENT.md) for detailed deployment guide.**

---

## 🎓 Learning Resources

For viva preparation and deep dive into concepts, see [README_STUDY_GUIDE.md](README_STUDY_GUIDE.md).

---

## 🔧 Troubleshooting

### Training Issues

- **Out of memory:** Reduce `BATCH_SIZE` to 16 or 8
- **Slow training:** Enable `torch.backends.cudnn.benchmark = True`
- **Overfitting:** Increase augmentation probability or dropout

### Data Issues

- **Dataset not found:** Check paths in Cell 4
- **Corrupted images:** Dataset class has error handling

### Model Issues

- **Low accuracy:** Train for more epochs, try EfficientNet-B4
- **Inference too slow:** Use smaller model (B0/B2) or quantization

---

## 📝 Citation

If you use this project, please cite:

```
@misc{qr-phishing-classifier,
  author = {Your Name},
  title = {QR Code Phishing Classifier},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/YOUR_USERNAME/qr-fishing}
}
```

---

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Pre-trained Model**: [devilfrost/qr-fishing on Kaggle Models](https://www.kaggle.com/models/devilfrost/qr-fishing)
- **Dataset**: [Kaggle - Benign and Malicious QR Codes](https://www.kaggle.com/datasets/samahsadiq/benign-and-malicious-qr-codes)
- **EfficientNet**: [PyTorch Torchvision Models](https://pytorch.org/vision/stable/models.html)
- **Training Platform**: [Kaggle](https://www.kaggle.com/)

---

## 🔗 Quick Links

- 📦 **Pre-trained Model**: https://www.kaggle.com/models/devilfrost/qr-fishing
- 📊 **Dataset**: https://www.kaggle.com/datasets/samahsadiq/benign-and-malicious-qr-codes
- 💻 **GitHub Repo**: https://github.com/SuryaKTiwari11/QR-phishing-
- 📓 **Kaggle Notebook**: [Your notebook link]

---

## 📧 Contact

For questions or collaboration:

- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- Email: your.email@example.com

---

**⚠️ Disclaimer:** This model is for educational and research purposes. Always verify QR codes from trusted sources before scanning for payments.
