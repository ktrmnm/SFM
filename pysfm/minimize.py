from ._core import _minimize


def minimize(F, method="fw", **kwargs):
    return _minimize(F, method, kwargs)
