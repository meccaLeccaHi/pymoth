# pyMoth
Python wrapper for [MothNet](https://github.com/charlesDelahunt/PuttingABugInML)

<img src='https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/Manduca_sexta_MHNT_CUT_2010_0_104_Dos_Amates_Catemaco_VeraCruz_Mexico_female_dorsal.jpg/519px-Manduca_sexta_MHNT_CUT_2010_0_104_Dos_Amates_Catemaco_VeraCruz_Mexico_female_dorsal.jpg' style="float: center; width: 50px">

#### Dependencies:
- numpy
- scipy
- matplotlib
- pillow
- [keras](https://keras.io/) (for loading MNIST)
- [dill](https://pypi.org/project/dill/)

Install manually with:
`$ pip install scipy matplotlib pillow keras dill`
-or-
`$ conda create -n <env_name> python=3.6 scipy matplotlib pillow keras dill`
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

[MNIST](http://yann.lecun.com/exdb/mnist/)