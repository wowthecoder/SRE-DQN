import sys
import importlib.util
from pathlib import Path

_canonical = Path(__file__).resolve().parent.parent / "dueling_double_dqn_sre.py"
_spec = importlib.util.spec_from_file_location("dueling_double_dqn_sre", _canonical)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["dueling_double_dqn_sre"] = _mod
_spec.loader.exec_module(_mod)
