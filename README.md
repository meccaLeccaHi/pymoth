# pyAL
Python wrapper for MothNet

#### Dependencies:
- numpy
- scipy
- matplotlib
- pillow
- [wget](https://pypi.org/project/wget/)

Install manually with:
`$ pip install numpy scipy matplotlib pillow wget`
Or, install from .txt file
1. clone this repository and `cd` into it
2a. `$ pip install -r requirements.txt` if you use pip, 
**or**
2b. `$ conda install --yes --file requirements.txt` if you prefer conda,
**or**
2c. `$ conda create --name <env_name> --file requirements.txt` to install into a conda virtualenv.

Uses [Decimal](https://docs.python.org/3/library/decimal.html) module* for floating point arithmetic to avoid representation errors (more info [here](https://docs.python.org/2/tutorial/floatingpoint.html)).
*Unlike hardware based binary floating point, the decimal module has an alterable precision (defaulting to 28 places). In this case, we rounded to four decimal digits.

[MNIST source](http://yann.lecun.com/exdb/mnist/)

[Matlab source](https://github.com/charlesDelahunt/PuttingABugInML)
