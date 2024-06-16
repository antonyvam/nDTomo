nDTomo software suite
=====================
The nDTomo software suite contains python scripts and GUIs for the simulation, visualisation and analysis of X-ray chemical imaging and tomography data.
The code includes both conventional algorithms as well as artifical neural networks developed using Pytorch.

Current status
--------------
Simulation code for creating phantoms (2D-4D)

Scripts for handling tomography data

Pytorch neural network models and related functions for various applications

nDTomoGUI for visualising chemical imaging and tomography data

To do
-----
Update the documentation

Installation instructions from sources
--------------------------------------
An important part of the code is based on astra-toolbox (and tomopy) which is currently available through conda so to make your life easier please install anaconda. It is possible to install astra-toolbox/tomopy from sources (i.e. if one wants to avoid using conda) but it is not a trivial task. I suggest you create a new anaconda environment for nDTomo (e.g. using the anaconda navigator) and make sure to install first the IDEs (Jupyter lab, Spyder etc) before installing the nDTomo.

To install from git:

```pip install --user git+https://github.com/antonyvam/nDTomo.git```

For development work:

```
git clone https://github.com/antonyvam/nDTomo.git && cd nDTomo
pip install --user -e .
```

For local installation, using the flag --user:

```
pip install --user -e .
pip install --user -r requirements.txt
```

or:

```
python3 setup.py install --user
pip install --user -r requirements.txt
```

For example, as a user at the Diamond Light Source:

```
git clone https://github.com/antonyvam/nDTomo.git && cd nDTomo
module load python/3
python setup.py install --user
pip install --user -r requirements.txt
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

Pytorch
----------------------------------------------

Neural networks were built/tested using Pytorch:

```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```

Other packages
----------------------------------------------
conda install defaults::conda-libmamba-solver
conda create --name finden_training -c conda-forge -c intel -c ccpi cil=23.0.1 astra-toolbox jupyterlab nb_conda_kernels "ipywidgets<8" --solver libmamba


Installation instructions using conda (Windows only)
----------------------------------------------------
The conda package of nDTomo might not be up to date so try installing from sources. The nDTomo is currently being developed using python v3.8

I suggest you create a new anaconda environment for nDTomo (e.g. using the anaconda navigator) and then simply run the following:

Make sure that you have added the following channels:

```conda config --add channels conda-forge```

```conda config --add channels astra-toolbox/label/dev```

Next install nDTomo:

```conda install -c antonyvam ndtomo```


Citation
--------
Please cite using the following:

Vamvakeros, A. et al., nDTomo software suite, 2019, DOI: https://doi.org/10.5281/zenodo.7139214, url: https://github.com/antonyvam/nDTomo

