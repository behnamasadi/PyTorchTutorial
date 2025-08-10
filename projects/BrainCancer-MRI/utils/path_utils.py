import os


def get_project_root() -> str:
    """
    Returns the absolute path to the root of the project directory.
    Assumes this file is located at: <project_root>/src/projects/BrainCancer-MRI/utils/path_utils.py
    """
    # Go up 4 levels: utils -> BrainCancer-MRI -> projects -> src -> root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
