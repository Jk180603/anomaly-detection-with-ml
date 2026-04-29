import sys
import os

sys.path.append(os.path.abspath("."))

from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Industrial Anomaly Detection API is running"