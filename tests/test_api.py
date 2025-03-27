# tests/test_api.py
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport # Import ASGITransport
from http import HTTPStatus
from io import BytesIO # For creating dummy file content

# Import app (try-except block remains the same)
try:
    from backend.fast import app as fastapi_app
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from backend.fast import app as fastapi_app


# Use ASGITransport fixture (scope="function")
@pytest_asyncio.fixture(scope="function")
async def async_client():
    """Provides an asynchronous test client for the FastAPI app using ASGITransport."""
    transport = ASGITransport(app=fastapi_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# --- Test Functions ---

@pytest.mark.asyncio
async def test_read_root(async_client: AsyncClient):
    """Test '/' endpoint."""
    response = await async_client.get("/")
    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"message": "Welcome to the Interpretable ML Dashboard API!"}

@pytest.mark.asyncio
async def test_get_dataset_summary_success(async_client: AsyncClient):
    """Test /dataset-summary/ success case."""
    response = await async_client.get("/dataset-summary/")
    assert response.status_code == HTTPStatus.OK
    data = response.json()
    # Add more specific assertions if desired, but keep basic checks for now
    assert data["dataset_name"] == "RarePlanes"
    assert data["data_directory_exists"] is True
    assert data["source_files_summary"]["train_images_archives_found"] == 1
    assert data["source_files_summary"]["test_annotations_archives_found"] == 1

@pytest.mark.asyncio
async def test_upload_dataset_placeholder(async_client: AsyncClient):
    """Test placeholder /upload-dataset/ endpoint."""
    dummy_file_content = b"This is dummy file content for testing."
    dummy_file_name = "test_upload_file.txt"
    # Prepare the file data in the format httpx expects for multipart/form-data
    files = {"uploaded_file": (dummy_file_name, BytesIO(dummy_file_content), "text/plain")}

    response = await async_client.post("/upload-dataset/", files=files)

    # Check status code
    assert response.status_code == HTTPStatus.OK
    # Check response content
    data = response.json()
    assert data["message"] == "File received successfully"
    assert data["filename"] == dummy_file_name
    assert data["content_type"] == "text/plain"

# Add tests for ML endpoint placeholders later...

