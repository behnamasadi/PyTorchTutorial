import os


def get_project_root() -> str:
    """
    Returns the absolute path to the root of the project directory.
    Assumes this file is located at: <project_root>/utils/path_utils.py
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
