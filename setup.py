#!/usr/bin/env python3

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mothnet",
    version="0.0.5",
    author="Adam Jones",
    author_email="ajones173@gmail.com",
    license='MIT',
    description="Neural network modeled after the olfactory system of the hawkmoth.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/meccaLeccaHi/pymoth",
    packages=['pymoth'],
    py_modules=[
        'pymoth.modules.classify',
        'pymoth.modules.generate',
        'pymoth.modules.params',
        'pymoth.modules.sde',
        'pymoth.modules.show_figs',
        'pymoth.MNIST_all.MNIST_make_all',
        # 'sample_experiment',
    ],
    install_requires=[
          'matplotlib',
          'scikit-learn',
          'scikit-image',
          'pillow',
          'keras',
          'tensorflow',
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
