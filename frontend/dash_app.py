# frontend/dash_app.py
import dash
from dash import html, dcc, Output, Input, State
import dash_bootstrap_components as dbc
import requests
import json
import base64
import io

# Import components
from frontend.components.summary_display import create_summary_display
from frontend.components.upload_component import create_upload_layout
from frontend.components.ml_controls import create_ml_controls # Import new component

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

    # Components from files
    create_summary_display(),
    create_upload_layout(),
    create_ml_controls(), # Add the ML controls layout here

    # Placeholder for future components (removing the old one)
    dbc.Row(dbc.Col(html.Hr(), width=12), className="mt-5"),

])

# --- Callbacks ---
# Callback for dataset summary (remains the same)
@app.callback(Output('dataset-summary-output', 'children'), Input('interval-component', 'n_intervals'))
def update_dataset_summary(n):
    # (Code remains the same...)
    try:
        api_url = f"{API_BASE_URL}/dataset-summary/"
        response = requests.get(api_url, timeout=10); response.raise_for_status()
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
    except Exception as e: return dbc.Alert(f"Error loading summary: {e}", color="warning")

# Callback for file upload (remains the same)
@app.callback(Output('upload-output', 'children'), Input('upload-data', 'contents'), State('upload-data', 'filename'), State('upload-data', 'last_modified'), prevent_initial_call=True)
def handle_upload(list_of_contents, list_of_names, list_of_dates):
     # (Code remains the same...)
     if list_of_contents is not None:
         content_type, content_string = list_of_contents.split(','); decoded = base64.b64decode(content_string); filename = list_of_names
         print(f"[Upload: {filename}] File selected.")
         try:
             files = {'uploaded_file': (filename, io.BytesIO(decoded), content_type.split(';')[0])}
             api_url = f"{API_BASE_URL}/upload-dataset/"
             print(f"[Upload: {filename}] Sending POST to {api_url}")
             response = requests.post(api_url, files=files, timeout=30); response.raise_for_status()
             backend_response = response.json()
             print(f"[Upload: {filename}] Backend response: {backend_response}")
             return dbc.Alert(f"Success: {backend_response.get('message', 'OK')} (Filename: {backend_response.get('filename', 'N/A')})", color="success")
         except Exception as e: print(f"[Upload: {filename}] Exception: {e}"); return dbc.Alert(f"Error processing file {filename}: {e}", color="danger")
     else: return ""

# --- NEW Callbacks for ML Buttons ---
@app.callback(Output('train-output', 'children'), Input('btn-train', 'n_clicks'), prevent_initial_call=True)
def handle_train_click(n_clicks):
    """Calls the backend /train-model/ placeholder endpoint."""
    if n_clicks:
        print("Train button clicked.")
        try:
            api_url = f"{API_BASE_URL}/train-model/"
            # Send empty JSON for now, params can be added later
            response = requests.post(api_url, json={}, timeout=60) # Longer timeout for potential training
            response.raise_for_status()
            backend_response = response.json()
            print(f"Train backend response: {backend_response}")
            return dbc.Alert(f"{backend_response.get('message', 'OK')}", color="info")
        except Exception as e:
            print(f"Train error: {e}")
            return dbc.Alert(f"Error calling train API: {e}", color="danger")
    return dash.no_update # Don't update if button wasn't clicked

@app.callback(Output('predict-output', 'children'), Input('btn-predict', 'n_clicks'), prevent_initial_call=True)
def handle_predict_click(n_clicks):
    """Calls the backend /predict/ placeholder endpoint."""
    if n_clicks:
        print("Predict button clicked.")
        try:
            api_url = f"{API_BASE_URL}/predict/"
            response = requests.post(api_url, json={}, timeout=30)
            response.raise_for_status()
            backend_response = response.json()
            print(f"Predict backend response: {backend_response}")
            return dbc.Alert(f"{backend_response.get('message', 'OK')}", color="info")
        except Exception as e:
            print(f"Predict error: {e}")
            return dbc.Alert(f"Error calling predict API: {e}", color="danger")
    return dash.no_update

@app.callback(Output('explain-output', 'children'), Input('btn-explain', 'n_clicks'), prevent_initial_call=True)
def handle_explain_click(n_clicks):
    """Calls the backend /explain/ placeholder endpoint."""
    if n_clicks:
        print("Explain button clicked.")
        try:
            api_url = f"{API_BASE_URL}/explain/"
            response = requests.post(api_url, json={}, timeout=30)
            response.raise_for_status()
            backend_response = response.json()
            print(f"Explain backend response: {backend_response}")
            return dbc.Alert(f"{backend_response.get('message', 'OK')}", color="info")
        except Exception as e:
            print(f"Explain error: {e}")
            return dbc.Alert(f"Error calling explain API: {e}", color="danger")
    return dash.no_update

# --- Main execution block (remains commented out) ---
# if __name__ == '__main__':
#     app.run(debug=True, port=8050)

