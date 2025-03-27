# Interpretable ML Dashboard

## Project Overview

This project aims to create a web-based dashboard for exploring and interpreting machine learning models trained on the RarePlanes dataset. It features a FastAPI backend for data processing, model training, prediction, and explainability (using techniques like SHAP and LIME), coupled with a Dash frontend for interactive visualization and control.

This project emphasizes best practices in software development, including version control (Git/GitHub), environment management (venv), automated testing, documentation, and CI/CD.

**Current Status (as of 2025-03-26):**
* Phase 1 (Initial Setup) Complete.
* Phase 2 (Backend Development) In Progress: API endpoints defined (placeholders for ML), basic logic for /dataset-summary/, initial tests added and passing.

## Technology Stack

* **Backend:** Python, FastAPI, Uvicorn
* **Frontend:** Dash, Plotly, Dash Bootstrap Components
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
4.  **Install dependencies (including test dependencies):**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

## Running the Backend Locally

To run the FastAPI backend server for development:

```bash
uvicorn backend.fast:app --reload --host 0.0.0.0 --port 8000
```

Access the API at `http://127.0.0.1:8000` and docs at `http://127.0.0.1:8000/docs`.

## Running Tests

This project uses `pytest` for backend API testing.

1.  **Ensure development dependencies are installed** (including `pytest` and `httpx`) by running `pip install -r requirements.txt`.
2.  **Run tests** from the project root directory:
    ```bash
    # Ensure venv is active
    venv/bin/python -m pytest -v
    # Or if 'python' reliably points to venv: python -m pytest -v
    ```

*(More sections like Frontend Usage, Deployment will be added later)*
