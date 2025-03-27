# frontend/dash_app.py
import dash
from dash import html, dcc, Output, Input
import dash_bootstrap_components as dbc
import requests
import json

# Import components
from frontend.components.summary_display import create_summary_display

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

# Define API base URL
API_BASE_URL = "http://127.0.0.1:8000"

# App layout using components
app.layout = dbc.Container([
    dcc.Interval(
        id='interval-component', interval=60*60*1000, n_intervals=0
    ),
    dbc.Row(dbc.Col(html.H1("Interpretable ML Dashboard", className="text-center my-4"))),
    create_summary_display(), # Uses the component function
    dbc.Row(dbc.Col(html.Hr(), width=12), className="mt-5"),
    dbc.Row(dbc.Col(html.Div("Future content area..."), width=12)),
])

# --- Callbacks ---
@app.callback(
    Output('dataset-summary-output', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_dataset_summary(n):
    """Fetches data from the /dataset-summary/ API endpoint and displays it nicely."""
    try:
        api_url = f"{API_BASE_URL}/dataset-summary/"
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        summary_data = response.json()

        list_items = [
            dbc.ListGroupItem(f"Dataset Name: {summary_data.get('dataset_name', 'N/A')}")
        ]
        summary_details = summary_data.get('source_files_summary', {})
        if summary_details:
             # --- Use html.H5 instead of deprecated ListGroupItemHeading ---
             list_items.append(html.H5("Source Archive Files Found:", className="mt-3 mb-2")) # Added some margin
             # --- End change ---
             for key, value in summary_details.items():
                  readable_key = key.replace('_', ' ').replace('archives found', '').strip().title()
                  list_items.append(dbc.ListGroupItem(f"{readable_key}: {value}"))
        formatted_output = dbc.ListGroup(list_items, flush=True, className="border-0") # Added border-0 for cleaner look maybe
        return formatted_output
    except requests.exceptions.Timeout: return dbc.Alert("Error: Timeout contacting API server.", color="danger")
    except requests.exceptions.ConnectionError: return dbc.Alert("Error: Could not connect to API server.", color="danger")
    except requests.exceptions.RequestException as e: return dbc.Alert(f"Error fetching data from API: {e}", color="danger")
    except Exception as e: return dbc.Alert(f"An error occurred: {e}", color="danger")

# --- Main execution block (remains commented out) ---
# if __name__ == '__main__':
#     app.run(debug=True, port=8050)

