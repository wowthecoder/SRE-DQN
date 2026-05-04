"""Process-wide lazy Gurobi environment.

Import _get_env() from here. All solver modules share one Env so licences are
acquired once, and output is suppressed globally.
"""

_ENV = None


def _get_env():
    global _ENV
    if _ENV is None:
        import gurobipy as gp
        _ENV = gp.Env(empty=True)
        _ENV.setParam("OutputFlag", 0)
        _ENV.setParam("LogToConsole", 0)
        _ENV.start()
    return _ENV
