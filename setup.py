import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='TrenchRipper',
    version='0.1.5',
    author="Daniel Eaton",
    packages=setuptools.find_packages(),
    install_requires=[
        'pandas==1.0.3',
        'numpy>=1.18.1',
        'bokeh==1.4.0',
        'h5py>=2.10.0',
        'h5py-cache==1.0',
        'scipy>=1.4.1',
        'scikit-image>=0.16.2',
        'jupyter-client==6.1.3',
        'jupyter-core==4.6.3',
        'jupyterlab==2.0.1',
        'jupyterlab-server==1.0.7',
        'jupyter-server-proxy==1.3.2',
        'matplotlib',
        'dask==2.12.0',
        'distributed==2.12.0',
        'dask-jobqueue==0.7.0',
        'dask-labextension==2.0.1',
        'tifffile>=2020.2.16',
        'ipywidgets==7.5.1',
        'pulp>=1.6.8',
        'fastparquet>=0.3.3',
        'torch>=1.4.0',
        'torchvision>=0.5.0',
        'tables>=3.6.1',
        'scikit-learn>=0.22.1',
        'seaborn==0.10.0',
        'h5py_cache==1.0',
        'nd2reader==3.2.3',
        'parse==1.15.0',
        'qgrid==1.3.1',
        'opencv-python'
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DanielScottEaton/TrenchRipper",
    python_requires='>=3.7',
)

#BROKEN
#     - bokeh==2.0.2
#     - chardet==3.0.4
#     - click==7.1.2
#     - cloudpickle==1.4.0
#     - distributed==2.15.1
#     - idna==2.9
#     - pexpect==4.8.0
#     - pytz==2020.1
#     - six==1.14.0
#     - typing-extensions==3.7.4.2

#WORKING
#     - bokeh==1.4.0
#     - click==7.1.1
#     - cloudpickle==1.3.0
#     - distributed==2.12.0
#     - importlib-metadata==1.6.0
#     - pytz==2019.3
#     - zipp==3.1.0



