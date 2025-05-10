# Overview

LightinGAN is a collection of a generation model using Pix2PixHD, and an object detection model YOLOv11 by Ultralytics. Our synthetic Data Generator creates images based on the input according to the type of scenario of the medical CT-Scan needed to be generated. Using this process, we have created very realistic images of the CT scan of the Lung Cancer for more feasibility and to improve our model for accurate predictions.

# Install

:warning: Please ensure you have suitable hardware and the supported OS to run the code. 

Supported OS: Windows, Linux, macOS.
Packages required to be installed: They are clearly defined in the requirements.txt file.
Hardware supported: Any graphics card that supports CUDA. RTX 20 series and above are recommended for inferences and training with fewer epochs.

**Command to install the recommended packages**
```bash
pip install -r requirements.txt
```


# Usage Example
## Generation model for generating images

ℹ️ Input: Image file of a normal CT-Scan, and the condition (0 - Benign, 1 - Malignant, 2 - Normal)

ℹ️ Output: Generated image with the scenario as given in the input.

### Sample Code
```python3
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from diffusers import AutoencoderKL

# ----- Your model classes -----
class SelfAttention(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.qkv = torch.nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.norm = torch.nn.LayerNorm([in_channels, 32, 32])  # Adjust size if needed
    
    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = q.view(B, C, H * W).permute(0, 2, 1)
        k = k.view(B, C, H * W)
        v = v.view(B, C, H * W).permute(0, 2, 1)
        attn = self.softmax(q @ k / C ** 0.5)
        out = (attn @ v).permute(0, 2, 1).view(B, C, H, W)
        return self.norm(out + x)

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = torch.nn.BatchNorm2d(out_channels)
        self.norm2 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.skip = torch.nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else torch.nn.Identity()
    def forward(self, x):
        identity = self.skip(x)
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.relu(x + identity)

class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.class_embedding = torch.nn.Linear(3, 4)
        self.encoder = torch.nn.Sequential(
            ResidualBlock(4, 64),
            SelfAttention(64),
            ResidualBlock(64, 128),
            SelfAttention(128),
            ResidualBlock(128, 256),
            SelfAttention(256),
            ResidualBlock(256, 512),
        )
        self.middle = SelfAttention(512)
        self.decoder = torch.nn.Sequential(
            ResidualBlock(512, 256),
            SelfAttention(256),
            ResidualBlock(256, 128),
            SelfAttention(128),
            ResidualBlock(128, 64),
            SelfAttention(64),
            torch.nn.Conv2d(64, 4, 3, padding=1)
        )
    def forward(self, x, class_cond):
        class_emb = self.class_embedding(class_cond).view(-1, 4, 1, 1)
        x = self.encoder(x + class_emb)
        x = self.middle(x)
        x = self.decoder(x)
        return x

# ----- Inference function -----
def run_inference(image_path, class_label_idx=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load image and preprocess
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)  # shape: [1, 3, 256, 256]

    # Load VAE and model
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    unet = UNet().to(device)
    unet.load_state_dict(torch.load("model.pth", map_location=device))
    unet.eval()
    vae.eval()

    with torch.no_grad():
        # Encode image to latent
        latent = vae.encode(x).latent_dist.sample()

        # Prepare class embedding (example: 0=benign, 1=malignant, 2=normal)
        class_onehot = torch.nn.functional.one_hot(torch.tensor([class_label_idx]), num_classes=3).float().to(device)

        # Run model
        out_latent = unet(latent, class_onehot)

        # Decode
        out_img = vae.decode(out_latent).sample

    # Save or show output
    save_image(out_img, "reconstructed.png")
    print("Saved reconstruction as 'reconstructed.png'")

if __name__ == "__main__":
    # Example: image, class 'benign' (index 0)
    run_inference("normal_image.jpg", class_label_idx=0)
```

## For Detection mode

You can run the below script as an inference.py file, and pass the image as
```bash
python inference.py path/to/test_image.jpg
```

### Sample code for inference
```python3
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import sys

# ----- Class Definition -----
class LungClassifier(nn.Module):
    def __init__(self):
        super(LungClassifier, self).__init__()
        self.base_model = timm.create_model("efficientnetv2_l", pretrained=False, num_classes=0)
        self.fc1 = nn.Linear(self.base_model.num_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 3)
    def forward(self, x):
        x = self.base_model(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class_names = ["benign", "malignant", "normal"]  # Index must match your training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image_path, model_path="model.pth"):
    # Load and prepare image
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)  # Shape: [1,3,512,512]

    # Model
    model = LungClassifier().to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        print(f"Prediction: {class_names[pred_idx]} (class index: {pred_idx})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py path/to/image.jpg")
        sys.exit(1)
    img_path = sys.argv[1]
    # You can set model_path manually here if not 'model.pth'
    predict(img_path, model_path="model.pth")
```


*For more information about the dataset, see:
Iraq-Oncology Teaching Hospital/National Center for Cancer Diseases (IQ-OTH/NCCD) lung cancer dataset [https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset].


# Citing LUCIDNet (for Generative model or Detection model)

If you use LUCIDNet, please cite the following work:

```LaTeX
@inproceedings{ctgan,
  title={Modeling Tabular data using Conditional GAN},
  author={Xu, Lei and Skoularidou, Maria and Cuesta-Infante, Alfredo and Veeramachaneni, Kalyan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```

---
