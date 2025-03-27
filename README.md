# Interpretable ML Dashboard

## Project Overview

This project aims to create a web-based dashboard for exploring and interpreting machine learning models trained on the RarePlanes dataset. It features a FastAPI backend for data processing, model training, prediction, and explainability (using techniques like SHAP and LIME), coupled with a Dash frontend for interactive visualization and control.

This project emphasizes best practices in software development, including version control (Git/GitHub), environment management (venv), automated testing, documentation, and CI/CD.

**Current Status (as of 2025-03-26):**
* Phase 1 (Initial Setup) Complete.
* Phase 2 (Backend Development) In Progress: Basic API structure with FastAPI setup. Root ('/') and placeholder ('/dataset-summary/') endpoints implemented.

## Technology Stack

* **Backend:** Python, FastAPI, Uvicorn
* **Frontend:** Dash, Plotly, Dash Bootstrap Components
* **ML:** TensorFlow (potentially others later)
* **Interpretability:** SHAP, LIME (planned)
* **Data Handling:** Pandas, NumPy
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
    # Pull LFS files (if not automatically downloaded on clone/checkout)
    git lfs pull
    ```
    *Note: The RarePlanes dataset files in `/data` are large and managed by Git LFS.*
3.  **Create and activate a Python virtual environment:**
    ```bash
    # Ensure you have a compatible Python version (e.g., 3.10+)
    python3 -m venv venv
    source venv/bin/activate # On macOS/Linux
    # On Windows: venv\Scripts\activate
    ```
4.  **Install dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

## Running the Backend Locally

To run the FastAPI backend server for development:

```bash
# Ensure your virtual environment is activated
uvicorn backend.fast:app --reload --host 0.0.0.0 --port 8000
```

You can access the API at `http://127.0.0.1:8000` and the interactive documentation (Swagger UI) at `http://127.0.0.1:8000/docs`.

*(More sections like Frontend Usage, Testing, Deployment will be added later)*
