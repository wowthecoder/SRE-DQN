import sys
import importlib.util
from pathlib import Path

_canonical = Path(__file__).resolve().parent.parent / "path_solver.py"
_spec = importlib.util.spec_from_file_location("path_solver", _canonical)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["path_solver"] = _mod
_spec.loader.exec_module(_mod)
