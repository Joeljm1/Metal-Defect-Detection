"""
Launcher for Streamlit Metal Defect Inspection Dashboard (Phase 6 / Task 7).
"""

import subprocess
import sys
from pathlib import Path


def main():
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    port = "8501"

    print("=" * 70)
    print("Launching Metal Surface Defect Inspection Dashboard (CSE411 Team 6)")
    print(f"App Entrypoint: {app_path}")
    print(f"Local URL: http://localhost:{port}")
    print("=" * 70)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        port,
        "--server.headless",
        "true",
    ]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[INFO] Dashboard stopped by operator.")


if __name__ == "__main__":
    main()
