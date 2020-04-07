from fracPy.ptyLab import DataLoader
from pathlib import Path


#dataFolder = Path(r'~/ptyLabData').expanduser()
#dataFolder.mkdir(exist_ok=True)
dataFolder = '/home/dbs660/ptyLabData'
obj = DataLoader(dataFolder)

obj.wavelength = 700e-9

obj.save(name='recent')
