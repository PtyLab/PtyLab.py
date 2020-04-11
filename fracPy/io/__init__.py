from fracPy.io import readHdf5
from pathlib import Path

def get_example_data_folder():
    """
    Returns a Path with the example data folder.
    """
    fracPy_folder = Path(__file__).parent.parent.parent
    return fracPy_folder / 'example_data'

if __name__ == '__main__':
    print(get_example_data_folder())