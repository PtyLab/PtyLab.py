from setuptools import setup

setup(
    name='fracpy',
    version='0.0',
    packages=['fracPy', 'fracPy.io', 'fracPy.utils', 'fracPy.config', 'fracPy.Operators'],
    url='',
    license='',
    author='Lars Loetgering, fracPy team',
    author_email='',
    description='', install_requires=['numpy', 'matplotlib', 'h5py', 'scipy', 'scikit-image',
                                      'tqdm']
)
