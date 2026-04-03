import asyncio
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_list_sop_files():
    response = client.get("/sop/")
    assert response.status_code == 200
    data = response.json()
    # We expect at least the dummy files we created
    assert "dummy.txt" in data
    assert "dummy.pdf" in data


def test_get_sop_txt_content():
    response = client.get("/sop/dummy.txt")
    assert response.status_code == 200
    json_data = response.json()
    # Should contain the content we wrote
    assert "Contoh SOP sederhana" in json_data["content"]


def test_get_sop_pdf_content():
    response = client.get("/sop/dummy.pdf")
    assert response.status_code == 200
    json_data = response.json()
    content = json_data["content"]
    # If PyMuPDF is not available, we get a placeholder message.
    assert "Sample PDF SOP" in content or "PDF content could not be extracted" in content
