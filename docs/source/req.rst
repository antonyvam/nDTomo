Installation of nDTomo software
-------------------------------

Repository
^^^^^^^^^^
The nDTomo project repository is available for download from GitHub: 
https://github.com/antonyvam/nDTomo

Installation instructions
^^^^^^^^^^^^^^^^^^^^^^^^^
An important part of the code is based on astra-toolbox (and tomopy) which is currently available through conda so to make your life easier please install anaconda. It is possible to install astra-toolbox/tomopy from sources (i.e. if one wants to avoid using conda) but it is not a trivial task. I suggest you create a new anaconda environment for nDTomo (e.g. using the anaconda navigator) and make sure to install first the IDEs (Jupyter lab, Spyder etc) before installing the nDTomo.

1. **Install nDTomo from GitHub**

Create a new environment:

```
conda create --name ndtomo python=3.9
conda activate ndtomo
```

To install from git:

```
pip install git+https://github.com/antonyvam/nDTomo.git
```

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

2. **Install pytorch**

For Windows/Linux with CUDA 11.8:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. **Install astra-toolbox**

```
conda install -c astra-toolbox -c nvidia astra-toolbox
