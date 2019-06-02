# pyMoth
Python wrapper for MothNet

#### Dependencies:
- numpy
- scipy
- matplotlib
- pillow
- [wget](https://pypi.org/project/wget/)
- [dill](https://pypi.org/project/dill/)

Install manually with:
`$ pip install scipy matplotlib pillow python-wget dill`
-or-
`$ conda create -n <env_name> python=3.6 scipy matplotlib keras dill`
pillow
Or, install from .txt file
1. clone this repository and `cd` into it
2a. `$ pip install -r requirements.txt` if you use pip, 
**or**
2b. `$ conda install --yes --file requirements.txt` if you prefer conda,
**or**
2c. `$ conda create --name <env_name> --file requirements.txt` to install a conda virtualenv.

Built for use with Unix systems - not tested in Windows.
- Requires Python 3

Uses [Decimal](https://docs.python.org/3/library/decimal.html) module* for floating point arithmetic to avoid representation errors (more info [here](https://docs.python.org/2/tutorial/floatingpoint.html)).
*Unlike hardware based binary floating point, the decimal module has an alterable precision (defaulting to 28 places). In this case, we rounded to four decimal digits.

[MNIST source](http://yann.lecun.com/exdb/mnist/)

[Matlab source](https://github.com/charlesDelahunt/PuttingABugInML)
