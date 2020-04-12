from setuptools import setup

setup(
    name='TrenchRipper',
    version='0.1',
    packages=['trenchripper',],
    install_requires=[
    'numpy','pandas','h5py','scipy','scikit-image',
    'jupyterlab','matplotlib','dask distributed',
    'dask-jobqueue','tifffile','ipywidgets','pytables',
    'scikit-learn','seaborn','line_profiler','h5py_cache',
    'nd2reader','nodejs','widgetsnbextension']
)