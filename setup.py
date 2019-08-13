#!/usr/bin/env python3

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="py_moth",
    version="0.0.1",
    author="Adam Jones",
    author_email="ajones173@gmail.com",
    description="Neural network modeled after the olfactory system of the hawkmoth",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/meccaLeccaHi/pymoth",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
    ],
)
