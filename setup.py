from setuptools import setup

setup(
    name="PtyLab",
    version="0.0.1",
    python_requires='>=3.9',
    packages=[
        "PtyLab",
        "PtyLab.io",
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
        "scikit-learn",
        "tqdm",
        "pyqtgraph",
        "cupy",
        "tables",
        "tensorflow-cpu"
    ],
)
