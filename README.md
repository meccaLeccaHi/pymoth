# pymoth

[![Build Status](https://travis-ci.org/meccaLeccaHi/pymoth.svg?branch=master)](https://travis-ci.org/meccaLeccaHi/pymoth)
[![Documentation Status](https://readthedocs.org/projects/pymoth/badge/?version=latest)](https://pymoth.readthedocs.io/?badge=latest)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

This package contains a Python version of [MothNet](https://github.com/charlesDelahunt/PuttingABugInML)

<img src='https://upload.wikimedia.org/wikipedia/commons/thumb/b/ba/Manduca_brasiliensis_MHNT_CUT_2010_0_12_Boca_de_Mato%2C_Cochoeiras_de_Macacu%2C_rio_de_Janeiro_blanc.jpg/320px-Manduca_brasiliensis_MHNT_CUT_2010_0_12_Boca_de_Mato%2C_Cochoeiras_de_Macacu%2C_rio_de_Janeiro_blanc.jpg'>

Neural network modeled after the olfactory system of the hawkmoth, _Manduca sexta_ (shown above).
> This repository contains a Python version of the code used in:
> - ["Putting a bug in ML: The moth olfactory network learns to read MNIST"](https://doi.org/10.1016/j.neunet.2019.05.012), _Neural Networks_ 2019

---
[Docs (via Sphinx)](https://pymoth.readthedocs.io/)
---

## Installation
Built for use with Mac/Linux systems - not tested in Windows.
- Requires Python 3.6+

### Via `pip`
```console
pip install mothnet
```

### From source
First, clone this repo and `cd` into it. Then run:
```console
# Install dependencies:  
pip install -r pymoth/docs/requirements.txt
# Run sample experiment:
python pymoth/examples.py
```

#### Dependencies (also see [`requirements.txt`](./docs/requirements.txt))
- [scipy](https://www.scipy.org/)
- [matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/)(for kNN and SVM models)
- [pillow](https://pillow.readthedocs.io/en/stable/)
- [keras](https://keras.io/) (for loading MNIST)
- [tensorflow](https://www.tensorflow.org/) (_also_ for loading MNIST)

---

### Example experiment (also see [`examples.py`](examples.py))
```python

import os
import pymoth

def experiment():

    # instantiate the MothNet object
    mothra = pymoth.MothNet({
        'screen_size': (1920, 1080), # screen size (width, height)
        'num_runs': 1, # how many runs you wish to do with this moth
        'goal': 15, # defines the moth's learning rates
        'tr_per_class': 1, # (try 3) the number of training samples per class
        'num_sniffs': 1, # (try 2) number of exposures each training sample
        'num_neighbors': 1, # optimization param for nearest neighbors
        'box_constraint': 1e1, # optimization parameter for svm
        'n_thumbnails': 1, # show N experiment inputs from each class
        'show_acc_plots': True, # True to plot, False to ignore
        'show_time_plots': True, # True to plot, False to ignore
        'show_roc_plots': True, # True to plot, False to ignore
        'results_folder': 'results', # string
        'results_filename': 'results', # will get the run number appended to it
        'data_folder': 'MNIST_all', # string
        'data_filename': 'MNIST_all', # string
                            })

    # loop through the number of simulations specified:
    for run in range(mothra.NUM_RUNS):

        # generate dataset
        feature_array = mothra.load_mnist()
        train_X, test_X, train_y, test_y = mothra.train_test_split(feature_array)

        # load parameters
        mothra.load_moth() # define moth model parameters
        mothra.load_exp() # define parameters of a time-evolution experiment

        # run simulation (SDE time-step evolution)
        sim_results = mothra.simulate(feature_array)
        # future: mothra.fit(X_train, y_train)

        # collect response statistics:
        # process the sim results to group EN responses by class and time
        EN_resp_trained = mothra.collect_stats(sim_results, mothra.experiment_params,
            mothra._class_labels, mothra.SHOW_TIME_PLOTS, mothra.SHOW_ACC_PLOTS,
            images_filename=mothra.RESULTS_FILENAME, images_folder=mothra.RESULTS_FOLDER,
            screen_size=mothra.SCREEN_SIZE)

        # reveal scores
        # score MothNet
        mothra.score_moth_on_MNIST(EN_resp_trained)
        # score KNN
        mothra.score_knn(train_X, train_y, test_X, test_y)
        # score SVM
        mothra.score_svm(train_X, train_y, test_X, test_y)

        # plot each model in a subplot of a single figure
        if mothra.SHOW_ROC_PLOTS:
            mothra.show_multi_roc(['MothNet', 'SVM', 'KNN'], mothra._class_labels,
            images_filename=mothra.RESULTS_FOLDER+os.sep+mothra.RESULTS_FILENAME+'_ROC_multi')
```

### Sample results
<img src='https://github.com/meccaLeccaHi/pymoth/blob/master/pymoth/results/results_ROC_multi_sample.png?raw=true'>

### Dataset
[MNIST Data](http://yann.lecun.com/exdb/mnist/)

### Modules
- [*classify.py*](https://github.com/meccaLeccaHi/pymoth/blob/master/pymoth/modules/classify.py) \
Classify output from MothNet model.
- [*generate.py*](https://github.com/meccaLeccaHi/pymoth/blob/master/pymoth/modules/generate.py) \
Download (if absent) and prepare down-sampled MNIST dataset.
- [*params.py*](https://github.com/meccaLeccaHi/pymoth/blob/master/pymoth/modules/params.py) \
Experiment and model parameters.
- [*sde.py*](https://github.com/meccaLeccaHi/pymoth/blob/master/pymoth/modules/sde.py) \
Run stochastic differential equation simulation.
- [*show_figs.py*](https://github.com/meccaLeccaHi/pymoth/blob/master/pymoth/modules/show_figs.py) \
Figure generation module.
- [*MNIST_make_all.py*](https://github.com/meccaLeccaHi/pymoth/blob/master/pymoth/MNIST_all/MNIST_make_all.py) \ Downloads and saves MNIST data to .npy file.

---

Questions, comments, criticisms? Feel free to drop us an [e-mail](
  mailto:ajones173@gmail.com?subject=pymoth)!


Bug reports, suggestions, or requests are also welcome! Feel free to [create an issue](
  https://github.com/meccaLeccaHi/pymoth/issues/new).  
