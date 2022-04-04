nDTomo software suite
=====================
The nDTomo software suite contains python scripts and GUIs for the simulation, visualisation and analysis of X-ray chemical imaging and tomography data

Documentation for the GUIs is provided at: https://ndtomo.readthedocs.io

Current status
--------------
Simulation code for creating phantoms (2D-4D)

Scripts for handling tomography data

Tensorflow neural network models and related functions

nDVis GUI for visualising chemical imaging and tomography data

To do
-----
Modernise the GUIs and update the documentation

Installation instructions
-------------------------
The nDTomo is currently being developed using python v3.8

In order to use all features, one has to use anaconda (e.g. for astra-toolbox). I suggest you create a new anaconda environment for nDTomo (e.g. using the anaconda navigator). Make sure to install first Spyder/Jupyter lab etc before installing the nDTomo.

To install from git:

```pip install git+https://github.com/antonyvam/nDTomo.git```

For development work:

```
git clone https://github.com/antonyvam/nDTomo.git && cd nDTomo
pip install -e .
```

For local installation, using the flag --user:

```
pip install --user -e .
```

or:

```
python3 setup.py install --user
```

For example, as a user at the Diamond Light Source:

```
git clone https://github.com/antonyvam/nDTomo.git && cd nDTomo
module load python/3
python setup.py install --user
```

The GUIs require PyQt5 but I have removed it from the setup.py because it breaks the spyder IDE. If you don't use anaconda/spyder (PyQt5 is included in anaconda), then you can install it with pip: 

```
pip install PyQt5
```

PyFAI
-----
The pyFAI version used in the nDTomo is 0.19

To run with GPU, you need to install pyopencl

For Windows, try installing Christoph Gohlke's repository: http://www.lfd.uci.edu/~gohlke/pythonlibs/

For example: 

```
pip install pyopencl-2021.2.10-cp38-cp38-win_amd64.whl
```

PyFAI installation instructions can be found here: http://www.silx.org/doc/pyFAI/dev/operations/index.html

Extra packages to be installed (not essential)
----------------------------------------------

Tomography:

```conda install -c astra-toolbox/label/dev astra-toolbox```

NeuralNets with Tensorflow (make sure to follow these instruction for GPU support: https://www.tensorflow.org/install/gpu):

```pip install tensorflow==2.7 tensorflow-addons==0.15```



Citation
--------
Please cite using the following:

Vamvakeros, A. et al., nDTomo software suite, 2019, DOI: https://doi.org/10.5281/zenodo.6344270

