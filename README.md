nDTomo software suite
=====================
The nDTomo software suite contains GUIs for the simulation, visualisation and analysis of X-ray chemical imaging and tomography data

Full documentation provided at: https://ndtomo.readthedocs.io

Tested with python v3.8

To install from git:
`pip install git+https://github.com/antonyvam/nDTomo.git`

For development work:
```
git clone https://github.com/antonyvam/nDTomo.git && cd nDTomo
pip install -e .
```

Extra packages to be installed:

Tomography:
`conda install -c astra-toolbox/label/dev astra-toolbox`
NeuralNets:
`pip install tensorflow==2.7 tensorflow-addons==0.15`
