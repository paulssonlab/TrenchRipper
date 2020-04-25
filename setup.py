import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='TrenchRipper',
    version='0.2',
    author="Daniel Eaton",
    packages=setuptools.find_packages(),
    install_requires=[
        'pandas==1.0.3',
        'numpy>=1.18.1',
        'h5py>=1.10.4',
        'scipy>=1.4.1',
        'scikit-image>=0.16.2',
        'jupyterlab==2.0.1',
        'matplotlib>=3.1.3',
        'dask==2.12.0',
        'distributed==2.12.0',
        'dask-jobqueue==0.7.0',
        'tifffile>=2020.2.16',
        'ipywidgets==7.5.1',
        'pulp>=1.6.8',
        'fastparquet>=0.3.3',
        'pytorch>=1.4.0',
        'torchvision>=0.5.0',
        'cudatoolkit==10.0.130',
        'pytables>=3.6.1',
        'scikit-learn>=0.22.1',
        'seaborn==0.10.0',
        'bokeh==1.4.0',
        'h5py_cache==1.0',
        'nd2reader==3.2.3',
        'parse==1.15.0',
        'qgrid==1.3.1',
        'opencv-python>=4.2.0.34'
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DanielScottEaton/TrenchRipper",
    python_requires='>=3.7',
)