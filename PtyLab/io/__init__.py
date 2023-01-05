from PtyLab.io import readHdf5
from pathlib import Path


def getExampleDataFolder():
    """
    Returns a Path with the example data folder.
    """
    PtyLab_folder = Path(__file__).parent.parent.parent
    return PtyLab_folder / "example_data"
