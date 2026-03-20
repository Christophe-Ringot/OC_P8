import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "src/dashboard/monitoring_dashboard.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])
