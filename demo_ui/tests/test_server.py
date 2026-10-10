import http.client
import importlib.util
import json
from pathlib import Path
import threading

import pytest


HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("demo_server", HERE / "server.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    instance = module.create_server(HERE.parent / "discovery_submission", port=0,
                                    runs=tmp_path_factory.mktemp("ui-runs"))
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield instance
    instance.shutdown()
    instance.app.close()
    instance.server_close()
    thread.join(timeout=5)


def request(server, path, method="GET", payload=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=30)
    body = json.dumps(payload) if payload is not None else None
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    data = response.read()
    result = json.loads(data) if "application/json" in response.getheader("Content-Type", "") else data
    status, returned_headers = response.status, dict(response.getheaders())
    connection.close()
    return status, result, returned_headers


def test_static_and_security_headers(server):
    status, body, headers = request(server, "/")
    assert status == 200 and b"run-form" in body
    assert "default-src 'self'" in headers["Content-Security-Policy"]
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert request(server, "/app.js")[0] == 200
    assert request(server, "/styles.css")[0] == 200


def test_bundled_studies_and_aligned_fields(server):
    status, result, _ = request(server, "/api/studies")
    assert status == 200
    assert {study["id"] for study in result["studies"]} == {"reference", "transport", "burgers"}
    for identifier in ("reference", "transport", "burgers"):
        status, detail, _ = request(server, f"/api/studies/{identifier}")
        assert status == 200 and detail["equation"]["terms"]
        status, field, _ = request(server, f"/api/studies/{identifier}/field")
        assert status == 200 and len(field["t"]) == len(field["prediction"]) == len(field["reference"])
        assert len(field["x"]) == len(field["prediction"][0])
        assert len(field["lower"]) == len(field["upper"]) == len(field["t"])


def test_evidence_and_complete_evaluation(server):
    status, result, _ = request(server, "/api/evidence")
    assert status == 200 and len(result["records"]) == 28
    assert result["graph"]["edges"]
    status, result, _ = request(server, "/api/research")
    assert status == 200 and len(result["benchmark"]["items"]) == 95
    assert result["verification"]["status"] == "passed"


@pytest.mark.parametrize("payload", [{"sensors": 3}, {"noise": -1}, {"seed": True}, {"seed": 2**32},
                                     {"example": "unknown"}, {"command": "echo"}, []])
def test_invalid_settings_do_not_start_a_process(server, payload):
    status, result, _ = request(server, "/api/runs", "POST", payload)
    assert status == 400 and result["error"]
    assert not server.app.jobs


def test_cross_origin_and_path_traversal_are_blocked(server):
    assert request(server, "/api/runs", "POST", {}, {"Origin": "https://example.com"})[0] == 403
    assert request(server, "/api/studies", headers={"Host": "example.com"})[0] == 403
    assert request(server, "/../discovery_submission/config.json")[0] == 404
    assert request(server, "/api/studies/reference/download/unknown")[0] == 404
    assert request(server, "/api/studies/missing")[0] == 404


def test_equation_download_matches_backend(server):
    status, result, headers = request(server, "/api/studies/reference/download/equation")
    assert status == 200
    assert "attachment" in headers["Content-Disposition"]
    assert result == module.read_json(HERE.parent / "discovery_submission/results/equation.json")
