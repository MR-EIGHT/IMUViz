import dash
from dash import Dash, html, dcc
import dash_uploader as du
import nibabel as nib
import numpy as np
import plotly.express as px
import os
from dash.dependencies import Input, Output, State
import glob

from uncertainty import *
from possibility import *
from plots import *
# from segmentation import *


app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Interactive MRI Uncertainty Visualizer"

UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

du.configure_upload(app, UPLOAD_FOLDER)

app.layout = html.Div([
    html.H1("Interactive MRI Uncertainty Visualizer", style={'textAlign': 'center'}),
    html.P("Upload, analyze, and visualize uncertainty in brain MRI images.",
           style={'textAlign': 'center', 'fontSize': '18px'}),

    du.Upload(
        id="upload-data",
        text="Drag and Drop or Click to Upload",
        max_files=1,
        filetypes=['nii'],
    ),
    
    html.Div(
        id='slice-slider-container',
        children=[
            html.Label("Select Z-Slice:", style={'fontSize': '20px', 'fontWeight': 'bold'}),
            dcc.Slider(
                id='slice-slider', 
                min=0, 
                max=10,
                step=1, 
                value=0, 
                tooltip={"placement": "bottom", "always_visible": True},
                marks={i: str(i) for i in range(0, 154, 10)},
            ),
            html.Div(
                id='hover-label',
                style={'textAlign': 'center', 'fontSize': '16px', 'marginTop': '10px'}
            )
        ],
        style={
            'width': '80%',
            'margin': '0 auto',
            'textAlign': 'center',
            'marginTop': '20px',
            'padding': '20px',
        }
    ),
    
html.Div(
    id='output-image-upload',
    style={
        'display': 'flex',               # Use flexbox for layout
        'justifyContent': 'center',      # Center horizontally
        'alignItems': 'center',          # Center vertically
        'margin': 'auto',                # Automatically set margins for centering
        'borderRadius': '10px',          # Rounded corners
    }
),
 # Panel for Outcome Explorer
    html.Div(
        id="outcome-explorer",
        children=[
            html.H3("Possible Outcomes", style={'textAlign': 'center'}),
            html.Div(
                id="outcomes-panel",
                style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'}
            )
        ],
        style={'width': '80%', 'margin': '20px'}
    ),

])

loaded_image_data = None

def parse_nifti_image(file_path):
    try:
        nifti_image = nib.load(file_path)
        data = nifti_image.get_fdata()
        return data
    except Exception as e:
        print(f"Error loading NIfTI image: {e}")
        return None

@app.callback(
    [Output("output-image-upload", "children", allow_duplicate=True),
     Output("slice-slider", "max"),
     Output("slice-slider", "value"),
     Output("slice-slider", "marks"),
     Output("hover-label", "children")],
    [Input("upload-data", "isCompleted"),
     Input("slice-slider", "value")],
    [State("upload-data", "fileNames")],
    prevent_initial_call=True
)
def handle_upload_and_slice(upload_complete, z_index, filenames):
    global loaded_image_data

    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # File upload handling
    if triggered_id == "upload-data" and upload_complete and filenames:
        file_pattern = os.path.join(UPLOAD_FOLDER, "**", filenames[0])
        file_path = glob.glob(file_pattern, recursive=True)
        if not file_path:
            return "Error: Uploaded file not found.", 10, 0, {}, ""

        file_path = file_path[0]

        try:
            loaded_image_data = parse_nifti_image(file_path)
            if loaded_image_data is None:
                return "Error processing the file.", 10, 0, {}, ""

            z_dim = loaded_image_data.shape[2]
            middle_slice = z_dim // 2

            slice_image = loaded_image_data[:, :, middle_slice]

            # Simulated uncertainty data (same dimensions as the slice)
            uncertainty_map = heuristic_uncertainty_detection(slice_image)


            fig = px.imshow(slice_image, color_continuous_scale='gray', title=f"Z-Slice: {middle_slice}")
            fig.update_layout(height=600, width=600, margin=dict(l=0, r=0, t=30, b=0))

            fig.update_traces(
            customdata=uncertainty_map,
            hovertemplate=(
                "X: %{x}<br>"
                "Y: %{y}<br>"
                "Intensity: %{z}<br>"
                "Uncertainty: %{customdata:.2f}<br>"
                "<extra></extra>"  # Removes default trace name
            ))

            marks = {i: str(i) for i in range(0, z_dim, max(1, z_dim // 20))}
            return dcc.Graph(id="graph-image",    config={
        "scrollZoom": True,  # Enable scroll-based zoom
        "displayModeBar": True,  # Show toolbar for zooming/panning
    },
 figure=fig), z_dim - 1, middle_slice, marks, f"Currently viewing Z-Slice: {middle_slice}"

        except Exception as e:
            return f"Error processing the file: {e}", 10, 0, {}, ""

    # Slice selection handling
    elif triggered_id == "slice-slider" and loaded_image_data is not None:
        slice_image = loaded_image_data[:, :, z_index]
        # Simulated uncertainty data (same dimensions as the slice)
        uncertainty_map = heuristic_uncertainty_detection(slice_image)
        fig = px.imshow(slice_image, color_continuous_scale='gray', title=f"Z-Slice: {z_index}")
        fig.update_layout(height=600, width=600, margin=dict(l=0, r=0, t=30, b=0))
        fig.update_traces(
            customdata=uncertainty_map,
            hovertemplate=(
                "X: %{x}<br>"
                "Y: %{y}<br>"
                "Intensity: %{z}<br>"
                "Uncertainty: %{customdata:.2f}<br>"
                "<extra></extra>"  # Removes default trace name
            ))
        return dcc.Graph(id="graph-image",     config={
        "scrollZoom": True,  # Enable scroll-based zoom
        "displayModeBar": True,  # Show toolbar for zooming/panning
    },
 figure=fig), dash.no_update, dash.no_update, dash.no_update, f"Currently viewing Z-Slice: {z_index}"

    return "No file uploaded yet.", 10, 0, {}, ""




@app.callback(
    Output("outcomes-panel", "children", allow_duplicate=True),
    [Input("graph-image", "relayoutData"), Input("slice-slider", "value")],
    prevent_initial_call=True
)
def show_possible_outcomes(relayout_data, z_index):
    global loaded_image_data

    # Ensure relayout_data and loaded_image_data are valid
    if relayout_data is None or loaded_image_data is None:
        return html.Div("No data to display outcomes for this zoomed region.")

    print("Relayout Data:", relayout_data)
    # Extract zoom coordinates from relayout_data
        # Extract ranges from relayout data
    try:
        x_min = int(relayout_data['xaxis.range[0]'])
        x_max = int(relayout_data['xaxis.range[1]'])
        y_min = int(relayout_data['yaxis.range[0]'])
        y_max = int(relayout_data['yaxis.range[1]'])

    except KeyError:
        return html.Div("Zoom into the image to see possible outcomes.")
    
        # Extract the region from the loaded data
    zoomed_region = loaded_image_data[min(y_min,y_max):max(y_min,y_max), min(x_min,x_max):max(x_min,x_max), z_index]

    # Generate multiple possible outcomes
    outcomes = []


    normalized_image = (zoomed_region - np.min(zoomed_region)) / (np.max(zoomed_region) - np.min(zoomed_region))

    # uncertainty_map = heuristic_uncertainty_detection(zoomed_region)
    intensity_map = detect_uncertainty_intensity_variation(normalized_image)
    gradient_map = detect_uncertainty_gradient(normalized_image)
    noise_map = detect_uncertainty_noise_blur(normalized_image)

    # Visualize results
    intensity_fig = visualize_intensity_variation(intensity_map)
    gradient_fig = visualize_gradient_uncertainty(gradient_map)
    noise_fig = visualize_noise_blur_uncertainty(noise_map)
    
    # Adjust layout for better visualization
    for fig in [intensity_fig, gradient_fig, noise_fig]:
        fig.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=30, b=0),  # Remove extra space
            xaxis_visible=True,
            yaxis_visible=True,
            height=300,
            width=300
        )

    outcomes.extend([
        dcc.Graph(figure=intensity_fig, style={"margin": "5px", "height": "300px", "width": "300px"}),
        dcc.Graph(figure=gradient_fig, style={"margin": "5px", "height": "300px", "width": "300px"}),
        dcc.Graph(figure=noise_fig, style={"margin": "5px", "height": "300px", "width": "300px"}),
    ])

    # Generate probabilistic overlay for uncertainty visualization
    uncertainty_map = (intensity_map + gradient_map + noise_map) / 3  # Example of combining maps
    uncertainty_map = np.clip(uncertainty_map, 0, 1)  # Ensure values are in [0, 1]

    # Overlay the uncertainty map on the zoomed region
    prob_overlay = overlay_prob_map(normalized_image, uncertainty_map)

    # Convert the overlay to a Plotly figure
    prob_overlay_fig = px.imshow(prob_overlay, title="Probabilistic Overlay", color_continuous_scale="viridis")
    prob_overlay_fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        height=300,
        width=300
    )


    # Append visualizations to outcomes
    outcomes.extend([
        dcc.Graph(figure=prob_overlay_fig, style={"margin": "5px", "height": "300px", "width": "300px"}),
    ])

    return html.Div(outcomes, style={"display": "flex", "flexWrap": "wrap", "justifyContent": "center"})




if __name__ == "__main__":
    app.run_server(debug=True)