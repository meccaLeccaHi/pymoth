# pymoth

[![Documentation Status](https://readthedocs.org/projects/pymoth/badge/?version=latest)](https://pymoth.readthedocs.io/?badge=latest)

This package contains a Python version of [MothNet](https://github.com/charlesDelahunt/PuttingABugInML)

<img src='https://upload.wikimedia.org/wikipedia/commons/thumb/b/ba/Manduca_brasiliensis_MHNT_CUT_2010_0_12_Boca_de_Mato%2C_Cochoeiras_de_Macacu%2C_rio_de_Janeiro_blanc.jpg/320px-Manduca_brasiliensis_MHNT_CUT_2010_0_12_Boca_de_Mato%2C_Cochoeiras_de_Macacu%2C_rio_de_Janeiro_blanc.jpg'>

Neural network modeled after the olfactory system of the hawkmoth, _Manduca sexta_ (shown above).
> This repository contains a Python version of the code used in:
> - ["Putting a bug in ML: The moth olfactory network learns to read MNIST"](https://doi.org/10.1016/j.neunet.2019.05.012), _Neural Networks_ 2019

-----
[docs](https://pymoth.readthedocs.io/)
-----

### Dependencies:
- [scipy](https://www.scipy.org/)
- [matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/)(for kNN and SVM models)
- [pillow](https://pillow.readthedocs.io/en/stable/)
- [keras](https://keras.io/) (for loading MNIST)

#### Install manually:  
> `$ pip install matplotlib scikit-learn scikit-image pillow keras`  

#### Or, install from [`requirements.txt`](./docs/requirements.txt) file:  
> `$ pip install -r requirements.txt`

### Run sample experiment:
`$ python experiment.py`

Built for use with Mac/Linux systems - not tested in Windows.
- Requires Python 3

### Data source:
[MNIST Data](http://yann.lecun.com/exdb/mnist/)

### Modules:
- *classify.py* Classify output from MothNet model.
- *generate.py* Download (if absent) and prepare down-sampled MNIST dataset.
- *params.py* Experiment and model parameters.
- *sde.py* Run stochastic differential equation simulation.
- *show_figs.py* Figure generation module.

### Sample results:
<img src='./results/results_ROC_multi_sample.png'>
