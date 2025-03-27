# tests/test_api.py
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from fastapi.testclient import TestClient
from http import HTTPStatus # For standard status codes

# Import the FastAPI app instance from your backend code
try:
    from backend.fast import app as fastapi_app
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from backend.fast import app as fastapi_app


# Pytest fixture to create an async test client for our app
@pytest_asyncio.fixture
async def async_client():
    """Create an async client for testing."""
    transport = ASGITransport(app=fastapi_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# Basic test function using the client fixture
@pytest.mark.asyncio
async def test_read_root(async_client):
    """Test if the root endpoint ('/') returns HTTP 200 OK."""
    try:
        response = await async_client.get("/")
        assert response.status_code == HTTPStatus.OK
    except Exception as e:
        pytest.fail(f"Test failed with exception: {str(e)}")

# Add more tests here later...

