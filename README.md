# Scientific Discovery Engine

`discovery_submission/` is the independent research codebase. It includes the complete source, generated datasets, trained models, benchmark outputs, scientific figures and technical report. Its README covers installation, scientific methods and reproduction.

`demo_ui/` is an optional local interface. The backend has no dependency on this folder. You can submit or run `discovery_submission/` on its own.

Install the backend dependencies first, following `discovery_submission/README.md`. With that Python environment active, start the interface from this folder:

```bash
python demo_ui/server.py
```

Open **http://127.0.0.1:8765** to access the workspace on local browser after running the script locally.
