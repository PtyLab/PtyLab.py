from pathlib import Path


def getExampleDataFolder():
    """
    Returns a Path with the example data folder, creating it if needed.
    """
    repo_folder = Path(__file__).parent.parent.parent
    folder = repo_folder / "example_data"
    # the folder holds only gitignored .hdf5 files, so a fresh clone lacks it
    folder.mkdir(exist_ok=True)
    return folder


from PtyLab.io.generateSimulationData import generate_simu_hdf5  # noqa: E402
