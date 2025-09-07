"""
Tests for file_utils module.
"""
import pytest
from pathlib import Path
import tempfile
import os
from unittest.mock import patch

from utility.file_utils import project_root, resource_path


class TestProjectRoot:
    """Test cases for project_root function."""

    def test_project_root_returns_path(self):
        """Test that project_root returns a Path object."""
        root = project_root()
        assert isinstance(root, Path)

    def test_project_root_is_absolute(self):
        """Test that project_root returns an absolute path."""
        root = project_root()
        assert root.is_absolute()

    def test_project_root_exists(self):
        """Test that the project root directory exists."""
        root = project_root()
        assert root.exists()
        assert root.is_dir()

    def test_project_root_consistency(self):
        """Test that project_root returns the same path on multiple calls."""
        root1 = project_root()
        root2 = project_root()
        assert root1 == root2

    def test_project_root_structure(self):
        """Test that project_root goes up one level from utils directory."""
        # The function should return the parent of the utils directory
        root = project_root()
        utils_dir = Path(__file__).resolve().parent.parent
        assert root == utils_dir


class TestResourcePath:
    """Test cases for resource_path function."""

    def test_resource_path_single_part(self):
        """Test resource_path with a single path component."""
        path = resource_path("config")
        expected = project_root() / "config"
        assert path == expected

    def test_resource_path_multiple_parts(self):
        """Test resource_path with multiple path components."""
        path = resource_path("config", "settings.yaml")
        expected = project_root() / "config" / "settings.yaml"
        assert path == expected

    def test_resource_path_nested_structure(self):
        """Test resource_path with deeply nested path components."""
        path = resource_path("data", "raw", "input", "file.txt")
        expected = project_root() / "data" / "raw" / "input" / "file.txt"
        assert path == expected

    def test_resource_path_returns_path_object(self):
        """Test that resource_path returns a Path object."""
        path = resource_path("test")
        assert isinstance(path, Path)

    def test_resource_path_is_absolute(self):
        """Test that resource_path returns an absolute path."""
        path = resource_path("test")
        assert path.is_absolute()

    def test_resource_path_empty_parts(self):
        """Test resource_path with no parts (should return project root)."""
        path = resource_path()
        assert path == project_root()

    def test_resource_path_with_empty_string(self):
        """Test resource_path with empty string parts."""
        path = resource_path("", "config", "", "file.txt")
        expected = project_root() / "" / "config" / "" / "file.txt"
        assert path == expected

    def test_resource_path_cross_platform(self):
        """Test that resource_path works correctly across platforms."""
        path = resource_path("folder", "subfolder", "file.txt")

        # The path should use the correct separator for the current OS
        path_str = str(path)
        if os.name == 'nt':  # Windows
            assert '\\' in path_str or '/' in path_str
        else:  # Unix-like systems
            assert '/' in path_str


class TestIntegration:
    """Integration tests for file_utils functions."""

    def test_project_root_and_resource_path_consistency(self):
        """Test that project_root and resource_path work together consistently."""
        root = project_root()
        resource = resource_path("test_file.txt")

        # resource_path should build on project_root
        assert resource.parent == root
        assert resource.name == "test_file.txt"

    def test_resource_path_relative_to_project_root(self):
        """Test that resource_path creates paths relative to project root."""
        root = project_root()
        config_path = resource_path("config", "app.yaml")

        # The config path should be relative to root
        relative_path = config_path.relative_to(root)
        assert str(relative_path) == os.path.join("config", "app.yaml")

    def test_example_from_docstring(self):
        """Test the specific example from the resource_path docstring."""
        cfg = resource_path("config", "config.yaml")
        expected = project_root() / "config" / "config.yaml"
        assert cfg == expected
        assert isinstance(cfg, Path)


class TestEdgeCases:
    """Test edge cases and potential error conditions."""

    def test_resource_path_with_special_characters(self):
        """Test resource_path with special characters in path components."""
        special_chars = ["file with spaces.txt",
                         "file-with-dashes.txt", "file_with_underscores.txt"]

        for filename in special_chars:
            path = resource_path("test", filename)
            assert path.name == filename
            assert isinstance(path, Path)

    def test_resource_path_with_dots(self):
        """Test resource_path with dots in path components."""
        path = resource_path("config", "app.config.yaml")
        assert path.name == "app.config.yaml"
        assert path.suffix == ".yaml"

    def test_project_root_logic_verification(self):
        """Test project_root logic by verifying the path structure."""
        # This test verifies that the project_root function follows the expected logic:
        # It should return the parent of the parent of the file_utils.py file

        # Get the actual file path of file_utils.py
        from utility import file_utils
        file_utils_path = Path(file_utils.__file__).resolve()

        # The project root should be two levels up from file_utils.py
        # file_utils.py -> utility/ -> utils/ -> project_root/
        expected_root = file_utils_path.parent.parent
        actual_root = project_root()

        assert actual_root == expected_root


if __name__ == "__main__":
    # Run tests if this file is executed directly
    pytest.main([__file__, "-v"])
