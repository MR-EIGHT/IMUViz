import numpy as np
import cv2
from dash import Dash, html, dcc

def generate_possibilities(image, uncertainty_map, num_outcomes=5):
    """
    Generate possible outcomes for uncertain regions in an image.

    Args:
        image (np.ndarray): Original MRI image.
        uncertainty_map (np.ndarray): Uncertainty values per pixel.
        num_outcomes (int): Number of possible outcomes to generate.

    Returns:
        list of np.ndarray: Generated images with variations in uncertain regions.
    """
    outcomes = []
    height, width = image.shape

    for _ in range(num_outcomes):
        modified_image = image.copy()

        # Generate variations in uncertain regions
        for y in range(height):
            for x in range(width):
                if uncertainty_map[y, x] > 0.5:  # Threshold for high uncertainty
                    noise = np.random.normal(0, uncertainty_map[y, x] * 10)  # Scale noise by uncertainty
                    modified_image[y, x] = np.clip(modified_image[y, x] + noise, 0, 255)

        outcomes.append(modified_image)

    return outcomes

def insert_synthetic_tumor(image, uncertainty_map, intensity_range=(200, 255)):
    """
    Insert synthetic tumors into high-uncertainty regions of the image.

    Args:
        image (np.ndarray): Original MRI image.
        uncertainty_map (np.ndarray): Uncertainty values per pixel.
        intensity_range (tuple): Intensity range for synthetic tumors.

    Returns:
        np.ndarray: Image with synthetic tumors.
    """
    modified_image = image.copy()
    height, width = image.shape

    # Threshold uncertainty map to identify uncertain regions
    uncertain_region = uncertainty_map > 0.5  # Define uncertain regions

    # Get indices of uncertain regions
    uncertain_indices = np.argwhere(uncertain_region)

    if uncertain_indices.shape[0] == 0:
        return modified_image  # No uncertain regions to insert tumors

    # Randomly select a center for the synthetic tumor from uncertain regions
    tumor_center = uncertain_indices[np.random.choice(uncertain_indices.shape[0])]

    tumor_radius = np.random.randint(5, 15)  # Random tumor radius

    # Draw tumor (ellipse) on the image
    cv2.ellipse(
        modified_image,
        (tumor_center[1], tumor_center[0]),
        (tumor_radius, tumor_radius),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=int(np.random.uniform(*intensity_range)),  # Random intensity within range
        thickness=-1  # Fill the ellipse
    )

    return modified_image


def monte_carlo_simulation(image, uncertainty_map, num_samples=5):
    """
    Perform Monte Carlo sampling to generate possible outcomes.
    
    Args:
        image (np.ndarray): Original MRI image.
        uncertainty_map (np.ndarray): Uncertainty values per pixel.
        num_samples (int): Number of samples to generate.
    
    Returns:
        list of np.ndarray: Generated images with Monte Carlo sampling.
    """
    outcomes = []
    for _ in range(num_samples):
        sampled_image = image.copy()
        
        # Apply random perturbations based on uncertainty
        noise = np.random.normal(0, uncertainty_map * 10, image.shape)  # Scale noise by uncertainty
        sampled_image = np.clip(sampled_image + noise, 0, 255)
        
        outcomes.append(sampled_image)
    
    return outcomes


import tensorflow_probability as tfp

def bayesian_tumor_generation(image, uncertainty_map):
    """
    Generate possible tumors using Bayesian inference.
    
    Args:
        image (np.ndarray): Original MRI image.
        uncertainty_map (np.ndarray): Uncertainty values per pixel.
    
    Returns:
        np.ndarray: Modified image with Bayesian-generated tumors.
    """
    prior = tfp.distributions.Normal(loc=0., scale=uncertainty_map)
    posterior = prior.sample()  # Simulated outcomes
    modified_image = image + posterior.numpy()
    return np.clip(modified_image, 0, 255)



def display_possibilities(possibilities):
    """
    Display generated possibilities in a grid layout.
    
    Args:
        possibilities (list of np.ndarray): List of generated images.
    
    Returns:
        html.Div: Dash layout with possibilities displayed.
    """
    figs = [
        dcc.Graph(
            figure=px.imshow(possibility, color_continuous_scale='Viridis'),
            style={"margin": "10px", "height": "300px", "width": "300px"}
        ) for possibility in possibilities
    ]
    return html.Div(figs, style={"display": "flex", "flexWrap": "wrap", "justifyContent": "center"})
