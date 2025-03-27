# frontend/dash_app.py
import dash
from dash import html, dcc, Output, Input # Added Output, Input
import dash_bootstrap_components as dbc
import requests
import json # To format the output nicely

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

# Define API base URL
API_BASE_URL = "http://127.0.0.1:8000" # FastAPI backend URL

# App layout
app.layout = dbc.Container([
    # Interval component to trigger callback on load (n_intervals=0)
    dcc.Interval(
        id='interval-component',
        interval=60*60*1000, # Update every hour (or choose a different interval)
        n_intervals=0 # Trigger once immediately on load
    ),
    # dcc.Store(id='api-data-store') # Optional: Store data if needed by multiple components

    dbc.Row(dbc.Col(html.H1("Interpretable ML Dashboard", className="text-center my-4"))),

    dbc.Row([
        dbc.Col([
            html.H3("Dataset Summary"),
            # Use dbc.Spinner for loading indicator while fetching
            dbc.Spinner(html.Div(id='dataset-summary-output', children="Loading summary..."))
        ], width=12)
    ]),

    # Placeholder for future components
    dbc.Row(dbc.Col(html.Hr(), width=12), className="mt-5"),
    dbc.Row(dbc.Col(html.Div("Future content area..."), width=12)),
])

# --- Callbacks ---
# Callback to fetch and display dataset summary
@app.callback(
    Output('dataset-summary-output', 'children'), # Update the Div content
    Input('interval-component', 'n_intervals') # Triggered by interval
)
def update_dataset_summary(n):
    """Fetches data from the /dataset-summary/ API endpoint and displays it."""
    try:
        api_url = f"{API_BASE_URL}/dataset-summary/"
        response = requests.get(api_url, timeout=10) # Add timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        summary_data = response.json()

        # Format the data using html.Pre for simple JSON display
        formatted_output = html.Pre(json.dumps(summary_data, indent=2))

        # Alternatively, format more nicely (Example):
        # card_content = []
        # card_content.append(html.P(f"Dataset: {summary_data.get('dataset_name', 'N/A')}"))
        # card_content.append(html.P(f"Data Dir Exists: {summary_data.get('data_directory_exists', 'N/A')}"))
        # summary = summary_data.get('source_files_summary', {})
        # for key, value in summary.items():
        #      card_content.append(html.P(f"{key.replace('_', ' ').title()}: {value}"))
        # formatted_output = dbc.Card(dbc.CardBody(card_content))

        return formatted_output

    except requests.exceptions.Timeout:
        return dbc.Alert("Error: Timeout contacting API server.", color="danger")
    except requests.exceptions.ConnectionError:
        return dbc.Alert("Error: Could not connect to API server. Is the backend running?", color="danger")
    except requests.exceptions.RequestException as e:
        return dbc.Alert(f"Error fetching data from API: {e}", color="danger")
    except Exception as e:
        return dbc.Alert(f"An error occurred: {e}", color="danger")


# --- Main execution block (remains commented out) ---
# if __name__ == '__main__':
#     app.run(debug=True, port=8050)

