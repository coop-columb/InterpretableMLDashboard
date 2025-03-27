# run_frontend.py
from frontend.dash_app import app

if __name__ == '__main__':
    # Use port 8050 for Dash app
    # Use app.run() instead of app.run_server() for newer Dash versions
    app.run(debug=True, port=8050)
