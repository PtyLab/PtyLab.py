from setuptools import setup, find_packages

setup(
    name='fracpy',
    version='0.0',
    packages=find_packages(),
    #packages=['fracPy', 'fracPy.io', 'fracPy.utils', 'fracPy.config', 'fracPy.Operators'],
    url='',
    license='',
    author='Lars Loetgering, fracPy team',
    author_email='',
    description='', install_requires=['numpy', 'matplotlib', 'h5py', 'scipy', 'scikit-image',
                                      'tqdm']
)
