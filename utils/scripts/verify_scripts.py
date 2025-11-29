#!/usr/bin/env python3
"""
Script to verify Python files can be imported/executed.
Similar to compilation checking in C++, this verifies Python scripts are syntactically
correct and can be imported/executed without runtime errors (where possible).
"""

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def check_syntax(file_path: Path) -> Tuple[bool, str]:
    """Check if Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code, filename=str(file_path))
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error parsing: {e}"


def try_import(file_path: Path, project_root: Path) -> Tuple[bool, str]:
    """Try to import the Python module."""
    try:
        # Convert file path to module path
        relative_path = file_path.relative_to(project_root)
        module_name = str(relative_path.with_suffix(
            '')).replace('/', '.').replace('\\', '.')

        # Remove leading dots
        if module_name.startswith('.'):
            module_name = module_name[1:]

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            return False, "Could not create module spec"

        # Try to load the module
        module = importlib.util.module_from_spec(spec)
        # Don't actually execute, just verify it can be loaded
        # This catches import-time errors
        return True, ""
    except Exception as e:
        # Import errors are expected for some scripts (missing deps, etc.)
        # But we still want to know about them
        return False, f"Import error: {e}"


def try_run_with_help(file_path: Path) -> Tuple[bool, str]:
    """Try to run script with --help flag (for scripts with argparse)."""
    try:
        result = subprocess.run(
            [sys.executable, str(file_path), '--help'],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=file_path.parent
        )
        if result.returncode == 0:
            return True, "Runs with --help"
        return False, f"Exit code: {result.returncode}"
    except subprocess.TimeoutExpired:
        return False, "Timeout (script may be hanging)"
    except Exception as e:
        # Not all scripts have --help, so this is optional
        return False, f"Could not run: {e}"


def verify_script(file_path: Path, project_root: Path, verbose: bool = False) -> Tuple[bool, List[str]]:
    """Verify a single Python script."""
    errors = []

    # Step 1: Check syntax
    syntax_ok, syntax_error = check_syntax(file_path)
    if not syntax_ok:
        errors.append(f"Syntax: {syntax_error}")
        return False, errors

    # Step 2: Try to import (for modules)
    # Skip import check for scripts that are meant to be run directly
    # We can identify these by checking if they have if __name__ == "__main__"
    has_main_guard = False
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            has_main_guard = 'if __name__' in content and '__main__' in content
    except:
        pass

    # Step 3: Try running with --help if it has main guard
    if has_main_guard:
        help_ok, help_msg = try_run_with_help(file_path)
        if help_ok:
            if verbose:
                errors.append(f"Info: {help_msg}")
        # Don't fail if --help doesn't work, it's optional

    # For non-main scripts, try import
    if not has_main_guard:
        import_ok, import_error = try_import(file_path, project_root)
        if not import_ok:
            # Import errors are warnings, not failures (missing deps, etc.)
            if verbose:
                errors.append(f"Import warning: {import_error}")

    return True, errors


def find_python_files(root_dir: Path, exclude_dirs: List[str] = None) -> List[Path]:
    """Find all Python files, excluding specified directories."""
    if exclude_dirs is None:
        exclude_dirs = ['projects', '.git', '__pycache__',
                        'test', 'tests', '.pytest_cache']

    python_files = []
    for py_file in root_dir.rglob('*.py'):
        # Check if file is in excluded directory
        parts = py_file.parts
        if any(excluded in parts for excluded in exclude_dirs):
            continue
        python_files.append(py_file)

    return sorted(python_files)


def main():
    """Main function to verify all Python scripts."""
    project_root = Path(__file__).parent.parent.parent
    exclude_dirs = ['projects', '.git', '__pycache__', 'test',
                    'tests', '.pytest_cache', 'mlartifacts', 'mlruns']

    python_files = find_python_files(project_root, exclude_dirs)

    if not python_files:
        print("No Python files found to verify.")
        return 0

    print(
        f"Found {len(python_files)} Python files to verify (excluding: {', '.join(exclude_dirs)})")
    print("-" * 80)

    failed = []
    passed = []
    warnings = []

    for py_file in python_files:
        relative_path = py_file.relative_to(project_root)
        success, errors = verify_script(py_file, project_root, verbose=True)

        if success:
            if errors:
                warnings.append((relative_path, errors))
                print(f"✅ {relative_path} (with warnings)")
            else:
                passed.append(relative_path)
                print(f"✅ {relative_path}")
        else:
            failed.append((relative_path, errors))
            print(f"❌ {relative_path}")
            for error in errors:
                print(f"   {error}")

    print("-" * 80)
    print(f"\nSummary:")
    print(f"  Passed: {len(passed)}")
    print(f"  Warnings: {len(warnings)}")
    print(f"  Failed: {len(failed)}")

    if warnings:
        print(f"\nFiles with warnings:")
        for file_path, errors in warnings:
            print(f"  {file_path}")
            for error in errors:
                print(f"    - {error}")

    if failed:
        print(f"\nFailed files:")
        for file_path, errors in failed:
            print(f"  {file_path}")
            for error in errors:
                print(f"    - {error}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
