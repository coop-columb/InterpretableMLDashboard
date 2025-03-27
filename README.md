# Interpretable ML Dashboard

## Project Overview

This project aims to create a web-based dashboard for exploring and interpreting machine learning models trained on the RarePlanes dataset. It features a FastAPI backend for data processing, model training, prediction, and explainability, coupled with a Dash frontend for interactive visualization and control.

This project emphasizes best practices in software development, including version control (Git/GitHub), environment management (venv), automated testing, documentation, and CI/CD.

**Current Status (as of 2025-03-26):**
* Phase 1 (Initial Setup) Complete.
* Phase 2 (Backend Development) Basic API structure and tests complete.
* Phase 3 (Frontend Development) In Progress: Basic Dash app layout created and running.

## Technology Stack

* **Backend:** Python, FastAPI, Uvicorn
* **Frontend:** Dash, Plotly, Dash Bootstrap Components
* **API Communication:** Requests
* **ML:** TensorFlow (potentially others later)
* **Interpretability:** SHAP, LIME (planned)
* **Data Handling:** Pandas, NumPy
* **Testing:** Pytest, HTTPX
* **Version Control:** Git, GitHub, Git LFS
* **Deployment (Planned):** Docker, Cloud Platform (e.g., AWS, Heroku)

## Setup

1.  **Clone the repository:**
    ```bash
    gh repo clone coop-columb/InterpretableMLDashboard
    cd InterpretableMLDashboard
    ```
2.  **Ensure Git LFS is installed:**
    ```bash
    git lfs install
    git lfs pull
    ```
3.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
4.  **Install dependencies (including test and frontend dependencies):**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

## Running Locally

You need to run the backend and frontend servers separately, typically in two different terminal windows/tabs (ensure the virtual environment is activated in both).

### Backend (FastAPI)

```bash
# Terminal 1 (Backend)
uvicorn backend.fast:app --reload --host 0.0.0.0 --port 8000
```
Access the API at `http://127.0.0.1:8000` and docs at `http://127.0.0.1:8000/docs`.

### Frontend (Dash)

```bash
# Terminal 2 (Frontend)
# Use app.run() for newer Dash versions
venv/bin/python run_frontend.py
```
Access the frontend dashboard in your browser at `http://127.0.0.1:8050`.

## Running Tests

This project uses `pytest` for backend API testing.

1.  **Ensure development dependencies are installed**.
2.  **Run tests** from the project root directory:
    ```bash
    venv/bin/python -m pytest -v
    ```

*(More sections like Deployment will be added later)*
