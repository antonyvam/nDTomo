![nDTomo](assets/ndtomo_logo_small.png)

nDTomo software suite
=====================
The nDTomo software suite contains python scripts and GUIs for the simulation, visualisation and analysis of X-ray chemical imaging and tomography data.
The code includes both conventional algorithms as well as artifical neural networks developed primarily using Tensorflow v2.

Documentation for the GUIs is provided at (needs to be updated): https://ndtomo.readthedocs.io

![XRD-CT](assets/xrdct.png)
Figure: Comparison between X-ray absorption-contrast CT (or microCT) and X-ray diffraction CT (XRD-CT or Powder diffraction CT) data acquired from an NMC532 Li ion battery. For more details regarding this study see reference 1.

Current status
--------------
Simulation code for creating phantoms (2D-4D)

Scripts for handling tomography data

Tensorflow neural network models and related functions for various applications

Clustering and dimensionality reduction methods (scikit-learn and autoencoders)

nDVis GUI for visualising chemical imaging and tomography data

To do
-----
Modernise the GUIs and update the documentation


Installation instructions from sources
--------------------------------------
An important part of the code is based on astra-toolbox (and tomopy) which is currently available through conda so to make your life easier please install anaconda. It is possible to install astra-toolbox/tomopy from sources (i.e. if one wants to avoid using conda) but it is not a trivial task. I suggest you create a new anaconda environment for nDTomo (e.g. using the anaconda navigator) and make sure to install first the IDEs (Jupyter lab, Spyder etc) before installing the nDTomo.

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

Neural networks were built/tested using Tensorflow; make sure to follow these instruction for GPU support: https://www.tensorflow.org/install/gpu. As an example:

```pip install tensorflow==2.7 tensorflow-addons==0.15```


Installation instructions using conda (Windows only)
----------------------------------------------------
The conda package of nDTomo might not be up to date so try installing from sources. The nDTomo is currently being developed using python v3.8

I suggest you create a new anaconda environment for nDTomo (e.g. using the anaconda navigator) and then simply run the following:

Make sure that you have added the following channels:

```conda config --add channels conda-forge```

```conda config --add channels astra-toolbox/label/dev```

Next install nDTomo:

```conda install -c antonyvam ndtomo```

Tensorflow is not included in the conda package so one has to install it using pip (please see below for more information).


Citation
--------
Please cite using the following:

Vamvakeros, A. et al., nDTomo software suite, 2019, DOI: https://doi.org/10.5281/zenodo.6344269, url: https://github.com/antonyvam/nDTomo


References
----------

1) A. Vamvakeros, D. Matras, T.E. Ashton, A.A. Coelho, H. Dong, D. Bauer, Y. Odarchenko, S.W.T. Price, K.T. Butler, O. Gutowski, A.‐C. Dippel, M. von Zimmerman, J.A. Darr, S.D.M. Jacques, A.M. Beale, Small Methods, 2100512, 2021