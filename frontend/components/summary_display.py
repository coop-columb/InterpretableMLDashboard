# frontend/components/summary_display.py
from dash import html
import dash_bootstrap_components as dbc

def create_summary_display():
    """Creates the layout section for displaying the dataset summary."""
    layout = dbc.Row([
        dbc.Col([
            html.H3("Dataset Summary"),
            dbc.Spinner(html.Div(id='dataset-summary-output', children="Loading summary..."))
        ], width=12)
    ])
    return layout

