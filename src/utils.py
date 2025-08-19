import os
import sys
import yaml
from datetime import datetime

# ------------------------
# Project root & path utilities
# ------------------------
def get_project_root():
    """
    Return the absolute path to the root of the project.
    Assumes this file is inside a subdirectory (e.g., src),
    so we go one directory up from its location.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_path_from_root(relative_path: str) -> str:
    """
    Build an absolute path starting from the project root.

    Args:
        relative_path (str): Path relative to the project root.
    Returns:
        str: Absolute path to the requested resource.
    """
    return os.path.join(get_project_root(), relative_path)

def get_outputs_dir(subdir: str = "") -> str:
    """
    Get (and create if necessary) the output directory path.
    The 'outputs' folder is placed alongside 'src' at the project root.

    Args:
        subdir (str): Optional subdirectory name within 'outputs'.
    Returns:
        str: Absolute path to the output directory.
    """
    outputs_dir = os.path.join(get_project_root(), "outputs", subdir)
    os.makedirs(outputs_dir, exist_ok=True)
    return outputs_dir

# ------------------------
# Config loading
# ------------------------
def load_config():
    """
    Load configuration from 'config.yaml' inside the 'src' folder.

    Returns:
        dict: Parsed YAML configuration.
    Raises:
        FileNotFoundError: If config.yaml does not exist.
    """
    config_path = os.path.join(get_project_root(), "src", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[ERROR] config.yaml not found at {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# ------------------------
# Console logging with colors
# ------------------------
class Logger:
    """
    Simple colored-console logger for info, warning, and error messages.
    Colors are ANSI escape codes; may not display correctly on all terminals.
    """
    COLORS = {
        "info": "\033[94m",     # Blue
        "warning": "\033[93m",  # Yellow
        "error": "\033[91m",    # Red
        "end": "\033[0m"        # Reset color
    }

    @classmethod
    def info(cls, msg):
        print(f"{cls.COLORS['info']}[INFO]{cls.COLORS['end']} {msg}")

    @classmethod
    def warning(cls, msg):
        print(f"{cls.COLORS['warning']}[WARNING]{cls.COLORS['end']} {msg}")

    @classmethod
    def error(cls, msg):
        print(f"{cls.COLORS['error']}[ERROR]{cls.COLORS['end']} {msg}")

# ------------------------
# Timestamped file naming
# ------------------------
def get_timestamped_filename(prefix: str, ext: str, subdir: str = "") -> str:
    """
    Generate a filename with a timestamp.

    Args:
        prefix (str): The file name prefix (without extension).
        ext (str): File extension (without dot).
        subdir (str): Optional subdirectory under 'outputs'.
    Returns:
        str: Full absolute path to the timestamped file.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(get_outputs_dir(subdir), f"{prefix}_{ts}.{ext}")
