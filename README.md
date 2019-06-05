# pyMoth
Python wrapper for [MothNet](https://github.com/charlesDelahunt/PuttingABugInML)

<img src='https://upload.wikimedia.org/wikipedia/commons/thumb/b/ba/Manduca_brasiliensis_MHNT_CUT_2010_0_12_Boca_de_Mato%2C_Cochoeiras_de_Macacu%2C_rio_de_Janeiro_blanc.jpg/320px-Manduca_brasiliensis_MHNT_CUT_2010_0_12_Boca_de_Mato%2C_Cochoeiras_de_Macacu%2C_rio_de_Janeiro_blanc.jpg'>

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
Or, install from .txt file  
> 1. clone this repository and `cd` into it  
> 2. 
> 	A. `$ pip install -r requirements.txt` if you use pip,   
> 	**or**  
> 	B. `$ conda install --yes --file requirements.txt` if you prefer conda,  
> 	**or**  
> 	C. `$ conda create --name <env_name> --file requirements.txt` to install a conda virtualenv.  

Built for use with Unix systems - not tested in Windows.
- Requires Python 3

[MNIST](http://yann.lecun.com/exdb/mnist/)