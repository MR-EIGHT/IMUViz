import numpy as np
import cv2
import scipy.ndimage

def heuristic_uncertainty_detection(image):
    # Normalize the image
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    
    # Calculate intensity variation (standard deviation)
    std_map = scipy.ndimage.generic_filter(normalized_image, np.std, size=5)
    
    # Calculate gradient magnitude
    sobel_x = scipy.ndimage.sobel(normalized_image, axis=0)
    sobel_y = scipy.ndimage.sobel(normalized_image, axis=1)
    gradient_map = np.hypot(sobel_x, sobel_y)
    
    # Combine heuristics (weight can be adjusted)
    uncertainty_map = 0.5 * std_map + 0.5 * gradient_map
    return uncertainty_map

import plotly.express as px

def visualize_uncertainty_map(uncertainty_map, title="Heuristic Uncertainty Heatmap"):
    fig = px.imshow(uncertainty_map, color_continuous_scale='Viridis', title=title)
    fig.update_layout(height=600, width=600, margin=dict(l=0, r=0, t=30, b=0))
    return fig


import numpy as np
import scipy.ndimage

def detect_uncertainty_intensity_variation(image, window_size=5):
    """
    Detect uncertainty using pixel intensity variation (std deviation).
    
    Args:
        image (np.ndarray): 2D input image.
        window_size (int): Size of the sliding window.
    
    Returns:
        np.ndarray: Uncertainty map based on intensity variation.
    """
    # Compute standard deviation in sliding windows
    uncertainty_map = scipy.ndimage.generic_filter(image, np.std, size=window_size)
    return uncertainty_map


import plotly.express as px

def visualize_intensity_variation(uncertainty_map, title="Intensity Variation Heatmap"):
    """
    Visualize uncertainty map from intensity variation.
    """
    fig = px.imshow(uncertainty_map, color_continuous_scale='Viridis', title=title)
    fig.update_layout(height=600, width=600, margin=dict(l=0, r=0, t=30, b=0))
    return fig




import scipy.ndimage

def detect_uncertainty_gradient(image):
    """
    Detect uncertainty using gradient-based analysis.
    
    Args:
        image (np.ndarray): 2D input image.
    
    Returns:
        np.ndarray: Uncertainty map based on gradient magnitudes.
    """
    # Compute gradients along x and y axes
    sobel_x = scipy.ndimage.sobel(image, axis=0)
    sobel_y = scipy.ndimage.sobel(image, axis=1)
    
    # Gradient magnitude
    gradient_magnitude = np.hypot(sobel_x, sobel_y)
    return gradient_magnitude

def visualize_gradient_uncertainty(gradient_map, title="Gradient Magnitude Heatmap"):
    """
    Visualize uncertainty map from gradient analysis.
    """
    fig = px.imshow(gradient_map, color_continuous_scale='Cividis', title=title)
    fig.update_layout(height=600, width=600, margin=dict(l=0, r=0, t=30, b=0))
    return fig


from skimage.filters.rank import entropy
from skimage.morphology import disk

def detect_uncertainty_noise_blur(image, neighborhood_radius=3):
    """
    Detect uncertainty using noise or blurriness estimation.
    
    Args:
        image (np.ndarray): 2D input image.
        neighborhood_radius (int): Radius for local entropy calculation.
    
    Returns:
        np.ndarray: Uncertainty map based on entropy.
    """
    # Compute local entropy to estimate uncertainty
    uncertainty_map = entropy(image, disk(neighborhood_radius))
    return uncertainty_map


def visualize_noise_blur_uncertainty(uncertainty_map, title="Noise/Blur Heatmap"):
    """
    Visualize uncertainty map from noise/blurriness analysis.
    """
    fig = px.imshow(uncertainty_map, color_continuous_scale='plasma', title=title)
    fig.update_layout(height=600, width=600, margin=dict(l=0, r=0, t=30, b=0))
    return fig





import numpy as np
import matplotlib.pyplot as plt
import cv2

def overlay_prob_map(image, prob_map, alpha=0.6, cmap='viridis'):
    """
    Overlay a probabilistic map on the original image.

    Args:
        image (np.ndarray): Grayscale MRI slice.
        prob_map (np.ndarray): Probability map with values between 0 and 1.
        alpha (float): Transparency level of the overlay.
        cmap (str): Colormap for the probability map.

    Returns:
        np.ndarray: Image with overlay.
    """
    # Normalize the image for display
    image_norm = (image - image.min()) / (image.max() - image.min())

    # Apply colormap to the probability map
    prob_colored = plt.cm.get_cmap(cmap)(prob_map)[:, :, :3]  # Remove alpha channel

    # Resize probability map to match image size
    prob_colored_resized = cv2.resize(prob_colored, (image.shape[1], image.shape[0]))

    # Combine image and probability map
    overlay = (1 - alpha) * image_norm[..., None] + alpha * prob_colored_resized
    overlay = np.clip(overlay, 0, 1)  # Ensure values are between 0 and 1
    return overlay
