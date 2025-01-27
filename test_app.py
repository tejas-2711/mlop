import pytest
from flask.testing import FlaskClient
import json
import sys
import os
from flask import render_template
from app import app

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             "../src")))


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_home(client: FlaskClient):
    """Test the home route"""
    response = client.get("/")
    assert response.status_code == 200
    # Decode response.data from bytes to a string
    response_data = response.data.decode('utf-8')
    # Render the template
    rendered_template = render_template("form.html")
    # Compare the decoded response data with the rendered template
    assert rendered_template in response_data


def test_predict(client: FlaskClient):
    """Test the predict route"""
    data = {"features": [3, 1, 22, 7.25, 0, 1, 2]}
    response = client.post(
        "/predict", data=json.dumps(data), content_type="application/json"
    )
    assert response.status_code == 200
    json_data = response.get_json()
    assert "prediction" in json_data