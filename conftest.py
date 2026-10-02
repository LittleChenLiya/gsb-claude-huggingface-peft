# Ensure the local src/ tree is importable when running tests from this worktree.
# This makes `python -m pytest` work without requiring an editable install or PYTHONPATH.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
