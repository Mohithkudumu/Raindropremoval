import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from model import UNetTransformer

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = UNetTransformer().to(device)
model.load_state_dict(torch.load("unet_transformer_best.pth", map_location=device))
model.eval()


def sliding_inference(model, image, tile=256, overlap=32):
    """
    Sliding window inference for high-resolution images.
    """
    B, C, H, W = image.shape
    stride = tile - overlap

    output = torch.zeros_like(image)
    count = torch.zeros_like(image)

    with torch.no_grad():
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                y_end = min(y + tile, H)
                x_end = min(x + tile, W)

                patch = image[:, :, y:y_end, x:x_end]

                # Pad smaller patches
                ph = tile - patch.shape[2]
                pw = tile - patch.shape[3]

                if ph > 0 or pw > 0:
                    patch = F.pad(patch, (0, pw, 0, ph), mode="reflect")

                pred = model(patch)

                # Remove padding
                pred = pred[:, :, :y_end - y, :x_end - x]

                output[:, :, y:y_end, x:x_end] += pred
                count[:, :, y:y_end, x:x_end] += 1

    return output / count


transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()


def predict(image):
    image = image.convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    pred = sliding_inference(model, img_tensor)
    pred = torch.clamp(pred, 0, 1)

    output_img = to_pil(pred.squeeze(0).cpu())

    return output_img


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Rainy Image"),
    outputs=gr.Image(type="pil", label="Cleaned Output"),
    title="🌧️ NTIRE Raindrop Removal",
    description="""
Upload a raindrop-degraded image captured in day or night conditions.
This model uses a Hybrid CNN-Transformer U-Net for image restoration.
""",
    allow_flagging="never"
)

demo.launch()