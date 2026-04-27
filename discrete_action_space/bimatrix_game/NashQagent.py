import sys
import importlib.util
from pathlib import Path

_canonical = Path(__file__).resolve().parent.parent / "NashQagent.py"
_spec = importlib.util.spec_from_file_location("NashQagent", _canonical)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["NashQagent"] = _mod
_spec.loader.exec_module(_mod)
