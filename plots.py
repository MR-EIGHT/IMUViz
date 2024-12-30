import plotly.express as px

def visualize_probabilistic_map(probability_map, title="Tumor Probability Map"):
    """
    Visualize the probabilistic map of uncertainty (tumor detection).
    
    Args:
        probability_map (np.ndarray): Probability values for each voxel.
        title (str): Title for the plot.
    
    Returns:
        plotly.graph_objs.Figure: The generated Plotly figure.
    """
    fig = px.imshow(probability_map, color_continuous_scale='RdBu_r', title=title)
    fig.update_layout(height=600, width=600, margin=dict(l=0, r=0, t=30, b=0))
    
    return fig


import plotly.graph_objects as go

def visualize_error_bars(tumor_volumes, error_margins, title="Tumor Volume with Uncertainty"):
    """
    Visualize tumor volume measurements with uncertainty using error bars.
    
    Args:
        tumor_volumes (list or np.ndarray): Measured tumor volumes.
        error_margins (list or np.ndarray): Uncertainty (error margin) for each tumor volume.
        title (str): Title for the plot.
    
    Returns:
        plotly.graph_objs.Figure: The generated Plotly figure.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(len(tumor_volumes)),
        y=tumor_volumes,
        error_y=dict(type='data', array=error_margins, visible=True),
        mode='markers+lines',
        name='Tumor Volume',
    ))
    fig.update_layout(title=title, xaxis_title="Sample", yaxis_title="Tumor Volume (cc)", height=600)
    return fig


def visualize_box_plot(data, title="Uncertainty Distribution in Tumor Measurement"):
    """
    Visualize uncertainty distribution using a box plot.
    
    Args:
        data (list or np.ndarray): Data points representing uncertainty (e.g., tumor volume across models).
        title (str): Title for the plot.
    
    Returns:
        plotly.graph_objs.Figure: The generated Plotly figure.
    """
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=data,
        boxmean='sd',
        name='Tumor Measurement Uncertainty',
    ))
    fig.update_layout(title=title, yaxis_title="Measurement Value", height=600)
    return fig


def visualize_uncertainty_histogram(uncertainty_values, title="Histogram of Uncertainty Values"):
    """
    Visualize the distribution of uncertainty values using a histogram.
    
    Args:
        uncertainty_values (list or np.ndarray): The uncertainty values to display.
        title (str): Title for the plot.
    
    Returns:
        plotly.graph_objs.Figure: The generated Plotly figure.
    """
    fig = px.histogram(uncertainty_values, nbins=20, title=title)
    fig.update_layout(xaxis_title="Uncertainty Value", yaxis_title="Frequency", height=600, width=600)
    return fig
