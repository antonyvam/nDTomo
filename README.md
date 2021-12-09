nDTomo software suite
=====================
The nDTomo software suite contains GUIs for the simulation, visualisation and analysis of X-ray chemical imaging and tomography data

Full documentation provided at: https://ndtomo.readthedocs.io

Installation instructions
-------------------------
The nDTomo is currently being developed using python v3.8

In order to use all features, one has to use anaconda (e.g. for astra-toolbox).

To install from git:

```pip install git+https://github.com/antonyvam/nDTomo.git```

For development work:

```
git clone https://github.com/antonyvam/nDTomo.git && cd nDTomo
pip install -e .
```

For a user at the Diamond Light Source:

```
git clone https://github.com/antonyvam/nDTomo.git && cd nDTomo
module load python/3
python setup.py install --user
```

Extra packages to be installed (not essential)
----------------------------------------------

Tomography:

```conda install -c astra-toolbox/label/dev astra-toolbox```

NeuralNets:

```pip install tensorflow==2.7 tensorflow-addons==0.15```

