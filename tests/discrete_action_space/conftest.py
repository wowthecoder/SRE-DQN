import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = REPO_ROOT / "discrete_action_space"

if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from path_solver import PathSolverWrapper


@pytest.fixture(scope="session")
def path_runtime_available():
    lib_path = MODULE_DIR / "pathwrap.so"
    if not lib_path.exists():
        pytest.skip(f"PATH wrapper library is missing: {lib_path}")

    try:
        solver = PathSolverWrapper(str(lib_path))
        solver.close()
    except Exception as exc:
        pytest.skip(
            "PATH runtime is unavailable. Configure PATH runtime (license/env/lib paths): "
            f"{exc}"
        )

    return str(lib_path)


@pytest.fixture(scope="session")
def path_solver(path_runtime_available):
    solver = PathSolverWrapper(path_runtime_available)
    yield solver
    solver.close()
