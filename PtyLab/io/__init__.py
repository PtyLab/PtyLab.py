from pathlib import Path


def getExampleDataFolder():
    """
    Returns a Path with the example data folder.
    """
    fracPy_folder = Path(__file__).parent.parent.parent
    return fracPy_folder / "example_data"


from PtyLab.io.generateSimulationData import generate_simu_hdf5  # noqa: E402
