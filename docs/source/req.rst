Installation of nDTomo software
-------------------------------

Repository
^^^^^^^^^^
The nDTomo project repository is available for download from GitHub: 
https://github.com/antonyvam/nDTomo

Installation instructions using conda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The nDTomo is currently being developed using python v3.8

In order to use all features, one has to use anaconda (e.g. for astra-toolbox). To install conda please see `Anaconda <https://www.anaconda.com/>`_. I suggest you create a new anaconda environment for nDTomo (e.g. using the anaconda navigator). 

```conda install -c antonyvam ndtomo```

Tensorflow is not included in the conda package so one has to install it using pip (please see below for more information).

Installation instructions from sources
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
An important part of the code is based on astra-toolbox (and tomopy) which is currently available through conda so to make your life easier please install anaconda. It is possible to install astra-toolbox/tomopy but I have not attempted it and not planning to anytime soon. I suggest you create a new anaconda environment for nDTomo (e.g. using the anaconda navigator) and make sure to install first Spyder/Jupyter lab etc before installing the nDTomo.

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

To install astra-toolbox:

```conda install -c astra-toolbox/label/dev astra-toolbox```

To install tomopy:

```conda install -c conda-forge tomopy```

The GUIs require PyQt5 but I have removed it from the setup.py because it breaks the spyder IDE. If you don't use anaconda/spyder (PyQt5 is included in anaconda), then you can install it with pip: 

```
pip install PyQt5
```

PyFAI
^^^^^
The pyFAI version used in the nDTomo is 0.19

To run with GPU, you need to install pyopencl

For Windows, try installing Christoph Gohlke's repository: http://www.lfd.uci.edu/~gohlke/pythonlibs/

For example: 

```
pip install pyopencl-2021.2.10-cp38-cp38-win_amd64.whl
```

PyFAI installation instructions can be found here: http://www.silx.org/doc/pyFAI/dev/operations/index.html

Extra packages to be installed (not essential)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NeuralNets with Tensorflow (make sure to follow these instruction for GPU support: https://www.tensorflow.org/install/gpu):

```pip install tensorflow==2.7 tensorflow-addons==0.15```
	
Dependencies
^^^^^^^^^^^^
nDTomo is a Python library which relies on the following packages:

* Python: version 3.8
* PyQt5
* Numpy
* Scipy
* Matplotlib
* Fabio
* Scikit-image
* h5py
* pyqtgraph
* pyFAI
* astra-toolbox


Contributors
^^^^^^^^^^^^

 * Antony Vamvakeros (Finden)
 * Simon Jacques (Finden)
 * Gavin Vaughan (ESRF)
 * Dorota Matras (Diamond Light Source)
 * Hongyang Dong (University College London)
 
Indirect contributors (ideas...)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

 * Marco di Michiel (ESRF)
 * Jérôme Kieffer (ESRF)
 
Beta testers
^^^^^^^^^^^^

 * Steve Price (Finden)
 * Donal Finegan (NREL)
 * Tom Heenan (University College London)



