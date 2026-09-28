"""Guard: the daily signals package must never drag in heavy dependencies.

The whole point of the product layer is that the daily job runs on a
lightweight install (see daily_signals/requirements.txt and the
signals-light CI job). Importing ray/tensorflow/torch/tensortrade at import
time would break that, so we assert — in a clean subprocess — that none of
them appear in sys.modules after importing the package and its CLI.
"""

import subprocess
import sys

CHECK = r"""
import sys
import daily_signals
import daily_signals.cli
import daily_signals.config

daily_signals.cli.build_parser()

forbidden = [name for name in ("ray", "tensorflow", "torch", "tensortrade")
             if name in sys.modules]
assert not forbidden, f"heavy modules imported at package import time: {forbidden}"
print("ok")
"""


def test_no_heavy_imports_at_package_import_time():
    result = subprocess.run(
        [sys.executable, "-c", CHECK],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
