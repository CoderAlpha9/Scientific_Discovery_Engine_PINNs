import argparse
import atexit
import json
import mimetypes
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from uuid import uuid4

import numpy as np


HERE = Path(__file__).resolve().parent
EXAMPLES = {"transport_reaction": "Transport and reaction", "transport": "Longer spatial domain", "burgers": "Nonlinear transport"}
DOWNLOADS = {"equation": "results/equation.json", "metrics": "results/metrics.json",
             "checkpoint": "results/models/inverse.pt", "observations": "data/observations.npz",
             "report": "technical_report.pdf"}


def read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


class Application:
    def __init__(self, backend, runs):
        self.backend, self.runs = Path(backend).resolve(), Path(runs).resolve()
        if not (self.backend / "discovery" / "__main__.py").is_file():
            raise ValueError("Backend folder must contain the discovery package.")
        self.jobs, self.lock = {}, threading.Lock()
        self.runs.mkdir(parents=True, exist_ok=True)

    def studies(self):
        found = {"reference": (self.backend, "Main research run", "bundled")}
        for name in ("transport", "burgers"):
            root = self.backend / "results" / "generalization" / name
            if (root / "results" / "metrics.json").is_file():
                found[name] = root, EXAMPLES[name], "validation"
        for root in sorted(self.runs.iterdir(), key=lambda path: path.name, reverse=True):
            if root.is_dir() and re.fullmatch(r"[a-f0-9]{12}", root.name) and (root / "results" / "metrics.json").is_file():
                data = read_json(root / "data" / "dataset.json", {})
                found[root.name] = root, f"{data.get('title', 'Experiment')} / seed {data.get('seed', '?')}", "live"
        return found

    def study(self, identifier):
        if identifier not in self.studies():
            raise FileNotFoundError("Experiment not found.")
        return self.studies()[identifier][0]

    def detail(self, identifier):
        root = self.study(identifier)
        result = {"id": identifier, "dataset": read_json(root / "data" / "dataset.json", {}),
                  "downloads": [key for key, name in DOWNLOADS.items() if (root / name).is_file()]}
        for key, name in (("equation", "equation"), ("hypotheses", "hypotheses"), ("candidates", "candidates"),
                          ("metrics", "metrics"), ("search", "search"), ("design", "experiment_design")):
            result[key] = read_json(root / "results" / f"{name}.json", {})
        return result

    def field(self, identifier):
        root = self.study(identifier)
        prediction = arrays(root / "results" / "frozen_prediction.npz")
        reference = arrays(root / "data" / "evaluation" / "reference.npz")
        observations = arrays(root / "data" / "observations.npz")
        row, column = slice(None, None, 2), slice(None, None, 2)
        result = {"x": prediction["x"][column].tolist(), "t": prediction["t"][row].tolist(),
                  "prediction": prediction["u"][row, column].tolist(), "reference": reference["u"][row, column].tolist(),
                  "length": float(observations["length"]), "observed_end": float(observations["time_end"]),
                  "observations": {key: value.tolist() for key, value in observations.items() if value.ndim == 1}}
        interval = root / "results" / "uncertainty.npz"
        if interval.exists():
            values = arrays(interval)
            result.update({key: values[key][row, column].tolist() for key in ("lower", "upper", "mean")})
        return result

    def start(self, payload):
        if not isinstance(payload, dict) or set(payload) - {"example", "sensors", "noise", "seed"}:
            raise ValueError("Unrecognized experiment settings.")
        example, sensors = payload.get("example", "transport_reaction"), payload.get("sensors", 16)
        noise, seed = payload.get("noise", 0.01), payload.get("seed", 11)
        if not isinstance(example, str) or example not in EXAMPLES:
            raise ValueError("Choose a supported experiment.")
        if type(sensors) is not int or sensors not in (8, 16, 24, 32):
            raise ValueError("Choose 8, 16, 24 or 32 sensors.")
        if type(noise) not in (int, float) or not np.isfinite(noise) or not 0 <= noise <= 0.1:
            raise ValueError("Noise must be between zero and 10 percent.")
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("Seed must be a nonnegative 32-bit integer.")
        with self.lock:
            if any(job["process"].poll() is None for job in self.jobs.values()):
                raise RuntimeError("An experiment is already running. Wait for it or cancel it first.")
            identifier = uuid4().hex[:12]
            directory = self.runs / identifier
            directory.mkdir()
            command = [sys.executable, "-u", "-m", "discovery", "experiment", "--offline",
                       "--root", str(self.backend), "--output", str(directory), "--example", example,
                       "--sensors", str(sensors), "--noise", str(noise), "--seed", str(seed)]
            environment = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2"}
            log = directory / "execution.log"
            with log.open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(command, cwd=self.backend, stdout=stream, stderr=subprocess.STDOUT,
                                           env=environment, shell=False)
            self.jobs[identifier] = {"id": identifier, "state": "running", "started": time.monotonic(),
                                     "process": process, "directory": directory, "log": log}
            threading.Thread(target=self.finish, args=(identifier,), daemon=True).start()
        return self.job(identifier)

    def finish(self, identifier):
        job = self.jobs[identifier]
        code = job["process"].wait()
        with self.lock:
            if job["state"] == "running":
                progress = read_json(job["directory"] / "results" / "progress.json", {})
                job["state"] = "completed" if code == 0 else ("inconclusive" if progress.get("state") == "inconclusive" else "failed")
            job["ended"] = time.monotonic()
            job["exit_code"] = code

    def job(self, identifier):
        if identifier not in self.jobs:
            raise FileNotFoundError("Job not found.")
        job = self.jobs[identifier]
        progress = read_json(job["directory"] / "results" / "progress.json", {})
        lines = job["log"].read_text(encoding="utf-8", errors="replace").splitlines()
        return {"id": identifier, "state": job["state"], "seconds": job.get("ended", time.monotonic()) - job["started"],
                "progress": progress, "log": lines[-12:], "study_id": identifier if job["state"] == "completed" else None}

    def cancel(self, identifier):
        with self.lock:
            if identifier not in self.jobs:
                raise FileNotFoundError("Job not found.")
            job = self.jobs[identifier]
            if job["process"].poll() is None:
                job["state"] = "cancelled"
                job["process"].terminate()
        return self.job(identifier)

    def close(self):
        for identifier in list(self.jobs):
            self.cancel(identifier)
        for job in self.jobs.values():
            try:
                job["process"].wait(timeout=5)
            except subprocess.TimeoutExpired:
                job["process"].kill()
                job["process"].wait()


def create_server(backend, port=8765, runs=None):
    app = Application(backend, runs or HERE / "runs")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *arguments):
            pass

        def send(self, content, content_type="application/json; charset=utf-8", status=200, filename=None):
            if not isinstance(content, bytes):
                content = json.dumps(content, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(content)

        def allowed(self):
            host = self.headers.get("Host", "")
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            origin = self.headers.get("Origin")
            return host in allowed and (origin is None or origin in {f"http://{value}" for value in allowed})

        def do_GET(self):
            try:
                if not self.allowed():
                    self.send({"error": "This interface accepts local requests only."}, status=403)
                    return
                path = urlsplit(self.path).path
                static = {"/": "index.html", "/app.js": "app.js", "/styles.css": "styles.css"}
                if path in static:
                    file = HERE / static[path]
                    self.send(file.read_bytes(), (mimetypes.guess_type(file.name)[0] or "text/plain") + "; charset=utf-8")
                elif path == "/favicon.ico":
                    self.send(b"", status=204)
                elif path == "/api/studies":
                    self.send({"studies": [{"id": key, "title": value[1], "kind": value[2]} for key, value in app.studies().items()],
                               "examples": [{"id": key, "title": value} for key, value in EXAMPLES.items()]})
                elif path == "/api/evidence":
                    records = [json.loads(line) for line in (app.backend / "data" / "evidence.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
                    self.send({"records": records, "graph": read_json(app.backend / "results" / "knowledge_graph.json", {})})
                elif path == "/api/research":
                    self.send({"benchmark": read_json(app.backend / "results" / "benchmark.json", {}),
                               "ablation": read_json(app.backend / "results" / "ablation_chain.json", {}),
                               "generalization": read_json(app.backend / "results" / "generalization.json", {}),
                               "verification": read_json(app.backend / "results" / "verification.json", {}),
                               "tests": read_json(app.backend / "results" / "tests.json", {})})
                elif match := re.fullmatch(r"/api/runs/([a-f0-9]{12})", path):
                    self.send(app.job(match[1]))
                elif match := re.fullmatch(r"/api/studies/([a-z0-9_]+)(?:/(field|download/([a-z]+)))?", path):
                    identifier, action, key = match.groups()
                    if action == "field":
                        self.send(app.field(identifier))
                    elif key:
                        if key not in DOWNLOADS:
                            raise FileNotFoundError("Artifact not found.")
                        file = app.study(identifier) / DOWNLOADS[key]
                        self.send(file.read_bytes(), mimetypes.guess_type(file.name)[0] or "application/octet-stream", filename=file.name)
                    else:
                        self.send(app.detail(identifier))
                else:
                    raise FileNotFoundError("Route not found.")
            except FileNotFoundError as error:
                self.send({"error": str(error)}, status=404)
            except (ValueError, KeyError) as error:
                self.send({"error": str(error)}, status=400)
            except (BrokenPipeError, ConnectionResetError):
                pass

            except Exception:
                self.send({"error": "Could not read the research artifacts. Run backend verification for details."}, status=500)

        def do_POST(self):
            try:
                if not self.allowed():
                    self.send({"error": "Cross-origin requests are not allowed."}, status=403)
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 4096:
                    raise ValueError("Request body is too large.")
                payload = json.loads(self.rfile.read(length) or b"{}")
                path = urlsplit(self.path).path
                if path == "/api/runs":
                    self.send(app.start(payload), status=202)
                elif match := re.fullmatch(r"/api/runs/([a-f0-9]{12})/cancel", path):
                    self.send(app.cancel(match[1]))
                elif path == "/api/verify":
                    run = subprocess.run([sys.executable, "-m", "discovery", "verify"], cwd=app.backend,
                                         capture_output=True, text=True, timeout=30)
                    self.send({"passed": run.returncode == 0,
                               "result": read_json(app.backend / "results" / "verification.json", {}) if run.returncode == 0 else {},
                               "message": "Artifact verification passed." if run.returncode == 0 else run.stderr.strip()})
                else:
                    raise FileNotFoundError("Route not found.")
            except (ValueError, TypeError) as error:
                self.send({"error": str(error)}, status=400)
            except FileNotFoundError as error:
                self.send({"error": str(error)}, status=404)
            except RuntimeError as error:
                self.send({"error": str(error)}, status=409)
            except subprocess.TimeoutExpired:
                self.send({"error": "Verification exceeded its time limit."}, status=504)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except OSError:
                self.send({"error": "Could not start the backend process. Check the Python environment and output folder."}, status=500)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.app = app
    return server


def main():
    parser = argparse.ArgumentParser(description="Optional local interface for the scientific discovery backend")
    parser.add_argument("--backend", type=Path, default=HERE.parent / "discovery_submission")
    parser.add_argument("--port", type=int, default=8765)
    arguments = parser.parse_args()
    server = create_server(arguments.backend, arguments.port)
    atexit.register(server.app.close)
    print(f"Discovery workbench: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.app.close()
        server.server_close()


if __name__ == "__main__":
    main()
