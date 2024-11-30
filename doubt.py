import dash
from dash import Dash, html, dcc, Input, Output, State
import dash_uploader as du
import nibabel as nib
import numpy as np
import plotly.express as px
import uuid
import os



app = dash.Dash(__name__)
app.title = "Interactive MRI Uncertainty Visualizer"  # Set the browser tab title

UPLOAD_FOLDER = "/tmp/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

du.configure_upload(app, UPLOAD_FOLDER)  # Configure dash_uploader


app.layout = html.Div([
    html.H1("Interactive MRI Uncertainty Visualizer", style={'textAlign': 'center'}),
    html.P("Upload, analyze, and visualize uncertainty in brain MRI images.", 
           style={'textAlign': 'center', 'fontSize': '18px'}), du.Upload( id="upload-data", text="Drag and Drop or Click to Upload", max_files=1, filetypes=['nii'],),
    html.Div(id='output-image-upload',
              style={
            'textAlign': 'center',  # Center-align text and content
            'marginTop': '20px'    # Add spacing between upload and output
        })])



# Function to parse and process the uploaded file
def parse_nifti_image(file_path):
    nifti_image = nib.load(file_path)
    data = nifti_image.get_fdata()  # 3D or 4D numpy array
    return data

# Callback for file upload
@du.callback(
    Output("output-image-upload", "children"),
    id="upload-data",
)
def display_uploaded_image(filenames):
    if not filenames:
        return "No file uploaded yet."
    
    # Construct the file path
    file_path = os.path.join(UPLOAD_FOLDER, filenames[0])
    
    # Process the uploaded NIfTI file
    try:
        image_data = parse_nifti_image(file_path)
        middle_slice = image_data.shape[2] // 2  # Select the middle slice
        slice_image = image_data[:, :, middle_slice]
        
        # Create a Plotly figure
        fig = px.imshow(slice_image, color_continuous_scale='gray', title="MRI Slice")
        fig.update_layout(height=600, width=600, margin=dict(l=0, r=0, t=30, b=0))
        
        return html.Div(
            dcc.Graph(figure=fig),
            style={'display': 'inline-block'}  # Center the image in the container
        )
    except Exception as e:
        return f"Error processing the file: {e}"

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
