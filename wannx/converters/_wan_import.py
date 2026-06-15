"""Helpers to locate WAN source package for converter imports."""

import os
import sys


def ensure_wan_import_path(checkpoint_dir: str) -> str:
    """Ensure a directory containing `wan/modules` is importable.

    Some checkpoints are weights-only (no local `wan` python package).
    We resolve `wan` from a small set of likely locations and prepend it
    to `sys.path` if needed.
    """
    this_file = os.path.abspath(__file__)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(this_file), "..", ".."))

    candidates = [
        os.path.abspath(os.path.join(checkpoint_dir, "..")),
        os.path.abspath(os.path.join(checkpoint_dir, "..", "..")),
        os.path.join(repo_root, "models", "wan2.1"),
    ]

    for root in candidates:
        if os.path.isfile(os.path.join(root, "wan", "modules", "model.py")):
            if root not in sys.path:
                sys.path.insert(0, root)
            return root

    checked = "\n".join(f"  - {c}" for c in candidates)
    raise ModuleNotFoundError(
        "Cannot locate WAN source package (`wan/modules/model.py`). "
        f"Searched:\n{checked}"
    )

