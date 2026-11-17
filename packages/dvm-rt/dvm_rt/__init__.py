"""SDK discovery without importing the device runtime. Python kernels use ``dvm``."""
from pathlib import Path


def get_library_dir():
    return str(Path(__file__).resolve().parent.parent / "dvm")


def get_include():
    return str(Path(get_library_dir()) / "include")
