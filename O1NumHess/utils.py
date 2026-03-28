import os
from pathlib import Path
from typing import Union


def getAbsPath(path: Union[str, Path]) -> Path:
    """Parse path, expand ``~``, make absolute, return normpath."""
    return Path(os.path.normpath(os.path.abspath(os.path.expanduser(path))))
