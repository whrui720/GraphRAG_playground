"""Launch the RAG Variants Visualizer.

Usage: python run_app.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
APP = ROOT / "app" / "streamlit_app.py"

if __name__ == "__main__":
    sys.exit(subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(APP), *sys.argv[1:]]
    ))
