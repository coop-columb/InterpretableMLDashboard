# frontend/components/upload_component.py
from dash import html, dcc
import dash_bootstrap_components as dbc

# Basic styling for the upload component (can be customized)
upload_style = {
    'width': '100%',
    'height': '60px',
    'lineHeight': '60px',
    'borderWidth': '1px',
    'borderStyle': 'dashed',
    'borderRadius': '5px',
    'textAlign': 'center',
    'margin': '10px'
}

def create_upload_layout():
    """Creates the layout section for file uploading."""
    layout = dbc.Row([
        dbc.Col([
            html.H3("Upload Data File"),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style=upload_style,
                # Allow multiple files to be uploaded? False for now
                multiple=False
            ),
            # Div to display upload status messages
            html.Div(id='upload-output', className="mt-2"),
        ], width=12)
    ], className="mt-4") # Add margin top
    return layout

