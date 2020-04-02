from fracPy.ptyLab import Ptylab

dataFolder = r'D:\ptyLab\ptyLabExport'

obj = Ptylab(dataFolder)

obj.wavelength = 700e-9

obj.save(name='recent')
