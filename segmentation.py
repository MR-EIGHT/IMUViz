import tensorflow as tf

# Load the trained U-Net model


from tensorflow.keras.layers import Conv2DTranspose

# Define a custom Conv2DTranspose builder function that takes all parameters
def build_custom_conv2d_transpose(**kwargs):
    return Conv2DTranspose(
        filters=kwargs.get('filters', 512),
        kernel_size=kwargs.get('kernel_size', (2, 2)),
        strides=kwargs.get('strides', (2, 2)),
        padding=kwargs.get('padding', 'same'),
        activation=kwargs.get('activation', 'linear'),
        use_bias=kwargs.get('use_bias', True),
        kernel_initializer=kwargs.get('kernel_initializer', 'glorot_uniform'),
        bias_initializer=kwargs.get('bias_initializer', 'zeros'),
        name=kwargs.get('name', None),  # Handle the 'name' argument
        trainable=kwargs.get('trainable', True),
        dtype=kwargs.get('dtype', 'float32')
    )

# Define the custom_objects dictionary
custom_objects = {
    'Conv2DTranspose': build_custom_conv2d_transpose
}

# Load the model with custom_objects
model = tf.keras.models.load_model('model.h5', custom_objects=custom_objects)




import numpy as np
import cv2

def preprocess_image(image):
    """
    Preprocess the input image for the U-Net model.
    
    Args:
        image (np.ndarray): 2D or 3D input image (grayscale or RGB).
        
    Returns:
        np.ndarray: Preprocessed image (resized and normalized).
    """
    # Resize the image to 256x256
    resized_image = cv2.resize(image, (256, 256))
    
    # Normalize the image (assuming the model was trained on [0, 1] scale)
    normalized_image = resized_image / 255.0

    # Add an extra dimension for batch size (1 image in the batch)
    normalized_image = np.expand_dims(normalized_image, axis=0)
    
    # If the image is grayscale, add a channel dimension (1 channel)
    if len(normalized_image.shape) == 3:  # For grayscale images
        normalized_image = np.expand_dims(normalized_image, axis=-1)
    
    return normalized_image


def make_prediction(image, model):
    """
    Make a prediction using the U-Net model.
    
    Args:
        image (np.ndarray): Preprocessed image ready for prediction.
        model (tf.keras.Model): The loaded U-Net model.
        
    Returns:
        np.ndarray: Segmentation mask (probability map).
    """
    # Predict the segmentation mask
    prediction = model.predict(image)
    
    # The model will likely return a 3D array (batch_size, 256, 256, 1), so we need to squeeze it
    prediction_mask = np.squeeze(prediction, axis=0)
    
    # Return the predicted mask (probability map)
    return prediction_mask


import plotly.express as px

def visualize_segmentation_overlay(image, prediction_mask, title="Tumor Segmentation"):
    """
    Visualize the original MRI image with the segmentation overlay.
    
    Args:
        image (np.ndarray): Original MRI image.
        prediction_mask (np.ndarray): Predicted tumor mask.
        title (str): Title for the plot.
    
    Returns:
        plotly.graph_objs.Figure: The generated Plotly figure with overlay.
    """
    # Normalize the original image
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # Create a heatmap overlay (tumor segmentation)
    overlay_image = np.ma.masked_where(prediction_mask < 0.5, prediction_mask)  # Mask out low probability regions

    # Create the visualization
    fig = px.imshow(normalized_image, color_continuous_scale='gray', title=title)
    fig.add_trace(px.imshow(overlay_image, color_continuous_scale='RdBu_r').data[0])
    
    # Adjust layout and appearance
    fig.update_layout(
        height=600,
        width=600,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False
    )
    
    return fig


