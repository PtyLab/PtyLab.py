from setuptools import setup

setup(
    name="PtyLab",
    version="0.0",
    packages=[
        "PtyLab",
        "PtyLab.read_write",
        "PtyLab.utils",
        "PtyLab.config",
        "PtyLab.Operators",
    ],
    url="",
    license="",
    author="Lars Loetgering, PtyLab team",
    author_email="",
    description="",
    install_requires=[
        "numpy",
        "matplotlib",
        "h5py",
        "scipy",
        "scikit-image",
        "tqdm",
        # 'pytables',
        "scikit-learn",
        # 'pyqt5',
        # 'napari[all]',
    ],
)
