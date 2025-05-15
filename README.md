![nDTomo](assets/ndtomo_logo_small.png)

nDTomo software suite
=====================
The nDTomo software suite contains a graphical user interface (GUI) and python scripts for the simulation, visualisation, pre-processing and analysis of X-ray chemical imaging and tomography data.

Documentation for the GUI will be provided at (under construction): https://ndtomo.readthedocs.io

The aim of this library, among others, is to generate tools for the following:
1. **Generation of multi-dimensional phantoms**
2. **Simulation of various pencil beam computed tomography data acquisition strategies**
3. **Processing and correcting sinogram data**
4. **Application of computed tomography reconstruction algorithms**
5. **Data analysis of chemical imaging data through peak fitting**

![XRD-CT](assets/xrdct.png)
Figure: Comparison between X-ray absorption-contrast CT (or microCT) and X-ray diffraction CT (XRD-CT or Powder diffraction CT) data acquired from an NMC532 Li ion battery. For more details regarding this study see [1].

Installation instructions
-------------------------
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
```


Citation
--------
If you use parts of the nDTomo code, please cite the work using the following:

Vamvakeros, A. et al., nDTomo software suite, 2019, DOI: https://doi.org/10.5281/zenodo.6344270, url: https://github.com/antonyvam/nDTomo


References
----------

[1] A. Vamvakeros, D. Matras, T.E. Ashton, A.A. Coelho, H. Dong, D. Bauer, Y. Odarchenko, S.W.T. Price, K.T. Butler, O. Gutowski, A.‐C. Dippel, M. von Zimmerman, J.A. Darr, S.D.M. Jacques, A.M. Beale, Small Methods, 2100512, 2021, https://doi.org/10.1002/smtd.202100512
