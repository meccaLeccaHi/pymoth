# pyAL
Python wrapper for MothNet

#### Dependencies:
- numpy
- scipy
-matplotlib
- pillow
- [wget](https://pypi.org/project/wget/) 

`$ pip install numpy scipy matplotlib pillow wget`

Uses [Decimal](https://docs.python.org/3/library/decimal.html) module* for floating point arithmetic to avoid representation errors (more info [here](https://docs.python.org/2/tutorial/floatingpoint.html)).
*Unlike hardware based binary floating point, the decimal module has an alterable precision (defaulting to 28 places). In this case, we rounded to four decimal digits.

[MNIST source](http://yann.lecun.com/exdb/mnist/)

[Matlab source](https://github.com/charlesDelahunt/PuttingABugInML)


