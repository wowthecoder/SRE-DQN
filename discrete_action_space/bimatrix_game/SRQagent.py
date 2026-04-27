import sys
import importlib.util
from pathlib import Path

_canonical = Path(__file__).resolve().parent.parent / "SRQagent.py"
_spec = importlib.util.spec_from_file_location("SRQagent", _canonical)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["SRQagent"] = _mod
_spec.loader.exec_module(_mod)
