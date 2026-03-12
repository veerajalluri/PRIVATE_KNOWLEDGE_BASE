import pytest
from fastapi.testclient import TestClient

pytest.importorskip("gradio", reason="UI dependencies not installed; run `poetry install --extras ui` to enable")


@pytest.mark.parametrize(
    "test_client", [{"ui": {"enabled": True, "path": "/ui"}}], indirect=True
)
def test_ui_starts_in_the_given_endpoint(test_client: TestClient) -> None:
    response = test_client.get("/ui")
    assert response.status_code == 200
