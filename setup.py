import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='TrenchRipper',
    version='0.2',
    author="Daniel Eaton",
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DanielScottEaton/TrenchRipper",
    python_requires='>=3.7',
)