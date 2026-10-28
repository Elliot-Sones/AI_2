"""Compatibility entry point for existing commands and serialized references.

The implementation lives in ai2.training.train.
"""
if __name__ == "__main__":
    import runpy
    runpy.run_module("ai2.training.train", run_name="__main__")
else:
    import sys
    from ai2.training import train as _implementation
    sys.modules[__name__] = _implementation
