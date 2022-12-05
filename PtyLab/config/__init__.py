# all the settings involving configuration go here
from pathlib import Path


def get_fracPy_folder():
    """Return the folder that PtyLab is installed in."""
    return Path(__file__).parent.parent
