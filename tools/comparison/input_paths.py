"""Only adapt the existing real-source test fixture path; keep assertions unchanged."""

import os
from pathlib import Path
from typing import Any


def pytest_collection_modifyitems(items: list[Any]) -> None:
    """Point the real fixture at the hash-checked source for this job."""
    for item in items:
        if item.module.__name__ == "test_mscheme3":
            item.module.REAL = Path(os.environ["QCHEM_W8_SOURCE"])
