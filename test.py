from monai.transforms import Compose, LoadImage, ScaleIntensity, Resize
import monai
print(monai.__version__)
# Define preprocessing pipeline
transforms = Compose([
    LoadImage(image_only=True),  # Load image
    ScaleIntensity(),           # Normalize intensity
    Resize((256, 256, 128))     # Resize to (H, W, D)
])

# Load and preprocess the image
image_path = "BraTS20_Training_001_t1.nii"
preprocessed_image = transforms(image_path)

import torch
import torch.nn.functional as F
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference

# Define your UNet model (2D example)
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,
    channels=(32, 64, 128, 256),
    strides=(2, 2, 2),
    num_res_units=2,
)

# Load model weights (if available)
# model.load_state_dict(torch.load('model_weights.pth'))

# Enable dropout during inference for MC Dropout
def mc_dropout_inference(model, image, num_samples=10):
    model.train()  # Set the model to training mode to enable dropout
    predictions = []
    for _ in range(num_samples):
        with torch.no_grad():
            output = model(image)
            predictions.append(output)
    # Calculate mean and uncertainty (variance)
    mean_prediction = torch.mean(torch.stack(predictions), dim=0)
    uncertainty = torch.var(torch.stack(predictions), dim=0)
    return mean_prediction, uncertainty

# Preprocess the image (ensure it's a tensor)
input_image = torch.tensor(preprocessed_image[None, None, :, :], dtype=torch.float32)

# Perform inference with MC Dropout
mean_prediction, uncertainty_map = mc_dropout_inference(model, input_image)


import matplotlib.pyplot as plt

# Visualize MRI slice with uncertainty overlay
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(preprocessed_image[:, :, 64], cmap='gray')
plt.title("Original MRI Slice")

plt.subplot(1, 2, 2)
plt.imshow(uncertainty_map[0, 0, :, :], cmap='hot', alpha=0.5)
plt.title("Uncertainty Map")
plt.colorbar()
plt.show()

