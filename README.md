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

Installation instructions from sources
--------------------------------------
The code has been developed and tested using Python v3.8
Fist install some required packages through conda:

```
conda install defaults::conda-libmamba-solver
conda install -c conda-forge -c intel -c ccpi pyfai=0.19 cil=23.0.1 astra-toolbox jupyterlab nb_conda_kernels "ipywidgets<8" --solver libmamba
```

Next, download the code from this github repository, unzip it, navigate with the terminal to the nDTomo directory where the setup.py file is located and run:

```
pip install --user -e .
pip install --user -r requirements.txt
```


Citation
--------
Please cite using the following:

A. Vamvakeros et al., nDTomo software suite, 2019, DOI: https://doi.org/10.5281/zenodo.7139214, url: https://github.com/antonyvam/nDTomo

