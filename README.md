# pyMoth
This package contains a Python version of [MothNet](https://github.com/charlesDelahunt/PuttingABugInML)

<img src='https://upload.wikimedia.org/wikipedia/commons/thumb/b/ba/Manduca_brasiliensis_MHNT_CUT_2010_0_12_Boca_de_Mato%2C_Cochoeiras_de_Macacu%2C_rio_de_Janeiro_blanc.jpg/320px-Manduca_brasiliensis_MHNT_CUT_2010_0_12_Boca_de_Mato%2C_Cochoeiras_de_Macacu%2C_rio_de_Janeiro_blanc.jpg'>

Neural network modeled after the olfactory system of the hawkmoth, _Manduca sexta_ (shown above).
> This repository contains a Python version of the code used in:
> - ["Putting a bug in ML: The moth olfactory network learns to read MNIST"](https://doi.org/10.1016/j.neunet.2019.05.012) (CB Delahunt and JN Kutz), _Neural Networks_ 2019

#### Dependencies:
- [scipy](https://www.scipy.org/)
- [matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/)(for kNN and SVM models)
- [pillow](https://pillow.readthedocs.io/en/stable/)
- [keras](https://keras.io/) (for loading MNIST)

#### Install manually:  
> `$ pip install scipy matplotlib scikit-learn pillow keras`  
> *-or-*  
> `$ conda create -n <env_name> python=3.6 scipy matplotlib scikit-learn pillow keras`  

#### Or, install from .txt file:  
> **First**,
> - clone this repository and `cd` into it  
>
> **Second**,
> - if you use pip, `$ pip install -r requirements.txt` *else,*   
> - if you prefer conda, `$ conda install --yes --file requirements.txt` *else,*
> - to install a conda virtualenv, `$ conda create --name <env_name> --file requirements.txt`  

#### Run via:
`$ python runMothLearnerMNIST.py`

Built for use with Mac/Linux systems - not tested in Windows.
- Requires Python 3

[MNIST Data](http://yann.lecun.com/exdb/mnist/)

#### Support modules:
- *classify.py* Classify model output for MNIST experiment.
- *extract.py* Prepare down-sampled digits from MNIST.
- *generate.py* Generate MNIST dataset (in the case that it's absent).
- *params.py* Experiment and model parameters.
- *sde.py* Run stochastic diff. equation simulation.
- *show_figs.py* Figure generation methods.

#### Sample results:
<img src='./results/results_ROC_multi_sample.png'>
