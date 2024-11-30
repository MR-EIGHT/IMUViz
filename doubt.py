import dash
from dash import Dash, html, dcc, Input, Output, State
import dash_uploader as du
import nibabel as nib
import numpy as np
import plotly.express as px


app = dash.Dash(__name__)
app.title = "Interactive MRI Uncertainty Visualizer"  # Set the browser tab title

du.configure_upload(app, '/tmp/uploads')  # Set upload folder

app.layout = html.Div([
    html.H1("Interactive MRI Uncertainty Visualizer", style={'textAlign': 'center'}),
    html.P("Upload, analyze, and visualize uncertainty in brain MRI images.", 
           style={'textAlign': 'center', 'fontSize': '18px'}), du.Upload( id="upload-data", text="Drag and Drop or Click to Upload", max_files=1, filetypes=['.gz'],),
    html.Div(id='output-image-upload')])




if __name__ == '__main__':
    app.run_server(debug=True)