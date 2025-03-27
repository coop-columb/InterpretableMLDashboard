# frontend/components/ml_controls.py
from dash import html, dcc
import dash_bootstrap_components as dbc

def create_ml_controls():
    """Creates the layout section for ML action buttons and outputs."""
    layout = dbc.Row([
        dbc.Col([
            html.H3("ML Actions"),
            dbc.ButtonGroup(
                [
                    dbc.Button("Train Model", id="btn-train", color="primary", className="me-1"),
                    dbc.Button("Predict", id="btn-predict", color="success", className="me-1"),
                    dbc.Button("Explain", id="btn-explain", color="info"),
                ],
                className="mb-2" # Margin bottom for spacing
            ),
            # Divs to display status/output for each action
            dbc.Spinner(html.Div(id='train-output', className="mt-2")),
            dbc.Spinner(html.Div(id='predict-output', className="mt-2")),
            dbc.Spinner(html.Div(id='explain-output', className="mt-2")),
        ], width=12)
    ], className="mt-4") # Add margin top
    return layout

