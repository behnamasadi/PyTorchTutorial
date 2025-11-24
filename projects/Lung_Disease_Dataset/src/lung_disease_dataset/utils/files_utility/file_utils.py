# utility/file_utils.py
"""
Legacy file_utils module - maintained for backward compatibility.

All path utilities have been moved to utils.paths.
This module re-exports from paths.py to maintain backward compatibility.
"""

from ..paths import project_root, resource_path, entry_script_dir

# Re-export all functions for backward compatibility
__all__ = ['project_root', 'resource_path', 'entry_script_dir']
