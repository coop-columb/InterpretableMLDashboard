# frontend/dash_app.py
import dash
from dash import html, dcc, Output, Input, State # Added State
import dash_bootstrap_components as dbc
import requests
import json
import base64 # Needed for decoding upload content
import io # Needed for sending decoded content as file

# Import components
from frontend.components.summary_display import create_summary_display
from frontend.components.upload_component import create_upload_layout # Import new component

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

    # Summary display component
    create_summary_display(),

    # Upload component
    create_upload_layout(), # Add the upload layout here

    # Placeholder for future components
    dbc.Row(dbc.Col(html.Hr(), width=12), className="mt-5"),
    dbc.Row(dbc.Col(html.Div("Future content area..."), width=12)),
])

# --- Callbacks ---
# Callback for dataset summary (remains the same)
@app.callback(
    Output('dataset-summary-output', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_dataset_summary(n):
    # (Code remains the same as before - fetching and formatting summary)
    try:
        api_url = f"{API_BASE_URL}/dataset-summary/"
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        summary_data = response.json()
        list_items = [ dbc.ListGroupItem(f"Dataset Name: {summary_data.get('dataset_name', 'N/A')}") ]
        summary_details = summary_data.get('source_files_summary', {})
        if summary_details:
             list_items.append(html.H5("Source Archive Files Found:", className="mt-3 mb-2"))
             for key, value in summary_details.items():
                  readable_key = key.replace('_', ' ').replace('archives found', '').strip().title()
                  list_items.append(dbc.ListGroupItem(f"{readable_key}: {value}"))
        formatted_output = dbc.ListGroup(list_items, flush=True, className="border-0")
        return formatted_output
    except Exception as e: # Simplified error handling for brevity
        return dbc.Alert(f"Error loading summary: {e}", color="warning")


# --- NEW Callback for file upload ---
@app.callback(
    Output('upload-output', 'children'), # Update the message div
    Input('upload-data', 'contents'), # Triggered when file is uploaded
    State('upload-data', 'filename'), # Get filename state
    State('upload-data', 'last_modified'), # Get timestamp state (optional)
    prevent_initial_call=True # Don't run on page load
)
def handle_upload(list_of_contents, list_of_names, list_of_dates):
    """Handles file uploads, sends file to backend API."""
    if list_of_contents is not None:
        # Assuming single file upload for now (multiple=False in dcc.Upload)
        content_type, content_string = list_of_contents.split(',')
        decoded = base64.b64decode(content_string)
        filename = list_of_names

        logger_prefix = f"[Upload: {filename}] " # Optional logger prefix
        print(logger_prefix + "File selected.") # Simple print logging for now

        try:
            # Prepare file data for requests
            files = {'uploaded_file': (filename, io.BytesIO(decoded), content_type.split(';') [0])} # Use io.BytesIO

            # Send file to backend API
            api_url = f"{API_BASE_URL}/upload-dataset/"
            print(logger_prefix + f"Sending POST to {api_url}")
            response = requests.post(api_url, files=files, timeout=30) # Increased timeout for upload
            response.raise_for_status() # Check for HTTP errors

            # Display backend response
            backend_response = response.json()
            print(logger_prefix + f"Backend response: {backend_response}")
            return dbc.Alert(f"Success: {backend_response.get('message', 'OK')} (Filename: {backend_response.get('filename', 'N/A')})", color="success")

        except requests.exceptions.Timeout:
            print(logger_prefix + "Timeout error.")
            return dbc.Alert("Error: Timeout connecting to backend API.", color="danger")
        except requests.exceptions.ConnectionError:
             print(logger_prefix + "Connection error.")
             return dbc.Alert("Error: Connection refused. Is the backend server running?", color="danger")
        except requests.exceptions.RequestException as e:
            print(logger_prefix + f"RequestException: {e}")
            return dbc.Alert(f"Error during upload: {e}", color="danger")
        except Exception as e:
            print(logger_prefix + f"Other exception: {e}")
            # Handle potential errors during base64 decoding or file processing
            return dbc.Alert(f"An error occurred processing the file: {e}", color="danger")
    else:
        # This part might not be strictly necessary with prevent_initial_call=True
        return "" # No file uploaded yet

# --- Main execution block (remains commented out) ---
# if __name__ == '__main__':
#     app.run(debug=True, port=8050)

