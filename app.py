import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr


# Model definition (copied from training notebook)
class QRClassifier(nn.Module):
    """EfficientNet with custom head for binary classification"""

    def __init__(
        self, model_name="efficientnet_b3", dropout_rate=0.3, hidden_units=256
    ):
        super(QRClassifier, self).__init__()
        PRETRAINED = False  # Do not load pretrained weights for inference
        if model_name == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=PRETRAINED)
        elif model_name == "efficientnet_b2":
            self.backbone = models.efficientnet_b2(pretrained=PRETRAINED)
        elif model_name == "efficientnet_b3":
            self.backbone = models.efficientnet_b3(pretrained=PRETRAINED)
        elif model_name == "efficientnet_b4":
            self.backbone = models.efficientnet_b4(pretrained=PRETRAINED)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(hidden_units, 1),
        )

    def forward(self, x):
        return self.backbone(x)


# Load model weights
MODEL_PATH = "best_model.pth"  # Make sure this file is uploaded to your Space
model = QRClassifier(model_name="efficientnet_b3", dropout_rate=0.3, hidden_units=256)
checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)
model.eval()


def predict(image):
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = image.convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()
    label = "🚨 Malicious" if prob >= 0.5 else "✅ Safe"
    confidence = max(prob, 1 - prob) * 100
    return {label: float(confidence)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload QR Code Image"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="QR Code Phishing Classifier",
    description="Upload a QR code image to detect if it's Safe or Malicious.",
)

if __name__ == "__main__":
    demo.launch()
