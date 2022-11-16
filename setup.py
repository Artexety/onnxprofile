from setuptools import setup, find_packages
from onnxprofile import __VERSION__, __AUTHOR__, __DESCRIPTION__

long_description = open("README.md", "r").read()
requirements = ["onnx", "numpy", "tabulate"]

setup(
    name="onnxprofile",                    
    version=__VERSION__,                   
    author=__AUTHOR__,                     
    description=__DESCRIPTION__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                     
    zip_safe=True,
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'onnxprofile = onnxprofile.__main__:main',
        ]
    },
)