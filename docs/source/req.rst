Installation of nDTomo software
-------------------------------

Repository
^^^^^^^^^^
The nDTomo project repository is available for download from GitHub: 
https://github.com/antonyvam/nDTomo

Installation instructions
^^^^^^^^^^^^^^^^^^^^^^^^^
An important part of the code is based on astra-toolbox (and tomopy), which is currently available through conda. To make your life easier, please install [Anaconda](https://www.anaconda.com/products/distribution). It is possible to install astra-toolbox/tomopy from sources (i.e., if one wants to avoid using conda), but it is not a trivial task. We recommend creating a new conda environment for `nDTomo`.

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

For development work (editable install):

```
git clone https://github.com/antonyvam/nDTomo.git && cd nDTomo
pip install -e .
```

Navigate to where the `setup.py` file is located and run:

```
pip install --user .
```

or:

```
python3 setup.py install --user
```

2. **Install pytorch**

For Windows/Linux with CUDA 11.8:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. **Install astra-toolbox**

```
conda install -c astra-toolbox -c nvidia astra-toolbox
```

4. **Launching the nDTomoGUI**

After installing `nDTomo`, the graphical user interface can be launched directly from the terminal:

```
conda activate ndtomo
nDTomoGUI
```

5. **Diamond Light Source**

As a user at the Diamond Light Source, you can install `nDTomo` by doing:

```
git clone https://github.com/antonyvam/nDTomo.git && cd nDTomo
module load python/3
python setup.py install --user
```


