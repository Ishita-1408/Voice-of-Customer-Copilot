import sys
import subprocess
from pathlib import Path


def main():
    """Launch the Streamlit Voice of Customer Copilot application."""
    app_path = Path(__file__).parent / "app" / "main.py"
    
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
    ] + sys.argv[1:]
    
    print(f"Launching Voice of Customer Copilot: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nApplication stopped.")


if __name__ == "__main__":
    main()
