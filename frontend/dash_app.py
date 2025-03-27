# frontend/dash_app.py
import dash
from dash import html, dcc # dcc needed for things like Interval, Store later
import dash_bootstrap_components as dbc
import requests # To make requests to the backend API

# Initialize the Dash app
# Use Dash Bootstrap Components theme for better styling
# Add suppress_callback_exceptions=True if callbacks are defined in separate files later
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server # Expose server for potential WSGI deployment later

# Define API base URL (running locally for now)
API_BASE_URL = "http://127.0.0.1:8000" # FastAPI backend URL

# App layout
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Interpretable ML Dashboard", className="text-center my-4"))),

    dbc.Row([
        dbc.Col([
            html.H3("Dataset Summary"),
            html.Div(id='dataset-summary-output', children="Loading summary..."),
            # We will use a callback later to fill this Div
        ], width=12)
    ]),

    # Placeholder for future components (visualizations, controls, etc.)
    dbc.Row(dbc.Col(html.Hr(), width=12), className="mt-5"),
    dbc.Row(dbc.Col(html.Div("Future content area..."), width=12)),

    # Interval component to trigger data loading (optional, for auto-refresh)
    # dcc.Interval(id='interval-component', interval=60*1000, n_intervals=0), # e.g., refresh every minute
    # dcc.Store(id='api-data-store') # To store data fetched from API
])

# --- Callbacks will go here later ---
# Example: Callback to fetch and display dataset summary
# @app.callback(
#     Output('dataset-summary-output', 'children'),
#     Input('interval-component', 'n_intervals') # Triggered by interval timer
# )
# def update_dataset_summary(n):
#     try:
#         response = requests.get(f"{API_BASE_URL}/dataset-summary/")
#         response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
#         summary_data = response.json()
#         # Format the data nicely using html components or dbc components
#         # Example: return html.Pre(json.dumps(summary_data, indent=2))
#         return f"Summary: {summary_data}" # Basic display for now
#     except requests.exceptions.RequestException as e:
#         return f"Error fetching data from API: {e}"
#     except Exception as e:
#         return f"An error occurred: {e}"


# --- Main execution block ---
# This allows running the Dash app directly via 
# However, we'll use a separate runner script for consistency.
# if __name__ == '__main__':
#     app.run_server(debug=True, port=8050)

