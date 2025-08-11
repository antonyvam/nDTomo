![nDTomo](assets/ndtomo_logo_small.png)

# nDTomo Software Suite

**nDTomo** is a Python-based software suite for the **simulation, visualization, pre-processing, reconstruction, and analysis** of chemical imaging and X-ray tomography data, with a focus on hyperspectral datasets such as X-ray powder diffraction computed tomography or XRD-CT.

It includes:

- A suite of **notebooks and scripts** for advanced processing, sinogram correction, CT reconstruction, peak fitting, and machine learning-based analysis
- A **PyQt-based graphical user interface (GUI)** for interactive exploration and analysis of hyperspectral tomography data
- A growing collection of **simulation tools** for generating phantoms and synthetic datasets

The software is designed to be accessible to both researchers and students working in chemical imaging, materials science, catalysis, battery research, and synchrotron radiation applications.

📘 Official documentation: https://ndtomo.readthedocs.io

![XRD-CT](https://raw.githubusercontent.com/antonyvam/nDTomo/master/assets/ndtomo_demo1.gif)

## Key Capabilities

nDTomo provides tools for:

1. **Interactive visualization of chemical tomography data** via the `nDTomoGUI`
2. **Generation of multi-dimensional synthetic phantoms**
3. **Simulation of pencil beam CT acquisition strategies**
4. **Pre-processing and correction of sinograms**
5. **CT image reconstruction** using algorithms like filtered back-projection and SIRT
6. **Dimensionality reduction and clustering** for unsupervised chemical phase analysis
7. **Pixel-wise peak fitting** using Gaussian, Lorentzian, and Pseudo-Voigt models
8. **Peak fitting using the self-supervised PeakFitCNN**
9. **Simultaneous peak fitting and tomographic reconstruction using the DLSR approach** with PyTorch GPU acceleration

![XRD-CT](assets/xrdct.png)

*Figure: Comparison between X-ray absorption-contrast CT (or microCT) and X-ray diffraction CT (XRD-CT or Powder diffraction CT) data acquired from a cylindrical Li-ion battery. For more details regarding these XRD-CT studies using cylindrical Li-ion batteries see [1,2].*

## Included Tutorials

The repository includes several **example notebooks** to help users learn the API and workflows:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/epapoutsellis/nDTomo/cloud_exs?urlpath=%2Fdoc%2Ftree%2Fdocs%2Fsource%2Fnotebooks) [![Tomography Reconstruction (GoogleColab)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/epapoutsellis/nDTomo/blob/cloud_exs/gcolab/tutorial_ct_recon_demo.ipynb)

| Notebook Filename | Topic |
|------------------|--------|
| [`tutorial_phantoms.ipynb`](docs/source/notebooks/tutorial_phantoms.ipynb) | Generating and visualizing 2D/3D phantoms |
| [`tutorial_pencil_beam.ipynb`](docs/source/notebooks/tutorial_pencil_beam.ipynb) | Simulating pencil beam CT data with different acquisition schemes |
| [`tutorial_detector_calibration.ipynb`](docs/source/notebooks/tutorial_detector_calibration.ipynb) | Calibrating detectors and integrating diffraction patterns using pyFAI |
| [`tutorial_texture_2D_diffraction_patterns.ipynb`](docs/source/notebooks/tutorial_texture_2D_diffraction_patterns.ipynb) | Investigating the effects of texture on 2D powder patterns |
| [`tutorial_sinogram_handling.ipynb`](docs/source/notebooks/tutorial_sinogram_handling.ipynb) | Pre-processing, normalization, and correction of sinograms |
| [`tutorial_ct_recon_demo.ipynb`](docs/source/notebooks/tutorial_ct_recon_demo.ipynb) | CT image reconstruction from sinograms using analytical and iterative methods |
| [`tutorial_dimensionality_reduction.ipynb`](docs/source/notebooks/tutorial_dimensionality_reduction.ipynb) | Unsupervised learning for phase identification in tomography |
| [`tutorial_peak_fitting.ipynb`](docs/source/notebooks/tutorial_peak_fitting.ipynb) | Peak fitting in synthetic XRD-CT datasets |
| [`tutorial_peak_fit_cnn.ipynb`](docs/source/notebooks/tutorial_peak_fit_cnn.ipynb) | Peak fitting in GPU using a self-supervised PeakFitCNN |
| [`tutorial_DLSR.ipynb`](docs/source/notebooks/tutorial_DLSR.ipynb) | Simultaneous peak fitting and CT reconstruction in GPU using the DLSR method |

Each notebook is designed to be **standalone and executable**, with detailed inline comments and example outputs.

**Note:**

- **Binder** is built with CPU-only support (including `torch`) and can be used to run all notebooks. However, some notebooks may take longer to execute due to the lack of GPU acceleration.

- **Google Colab** provides GPU support and `torch` is preinstalled. You will also need to install `nDTomo` at the beginning of each notebook session.

## Graphical User Interface (nDTomoGUI)

The `nDTomoGUI` provides a complete graphical environment for:

- Loading `.h5` / `.hdf5` chemical imaging datasets
- Visualizing 2D slices and 1D spectra interactively
- Segmenting datasets using channel selection and thresholding
- Extracting and exporting local diffraction patterns
- Performing single-peak batch fitting across regions of interest
- Generating a synthetic XRD-CT phantoms for development tests
- Using an embedded IPython console for advanced control and debugging

The GUI is described in more detail in the [online documentation](https://ndtomo.readthedocs.io) and supports both novice and expert workflows.

Launch with:

```bash
conda activate ndtomo
python -m nDTomo.gui.nDTomoGUI
```

## Installation Instructions

To make your life easier, please install [Anaconda](https://www.anaconda.com/products/distribution). The `nDTomo` library and all associated ode can be installed by following the next three steps:

### 1. Install astra-toolbox

An important part of the code is based on astra-toolbox, which is currently available through conda.

It is possible to install astra-toolbox from sources (i.e., if one wants to avoid using conda), but it is not a trivial task. We recommend creating a new conda environment for `nDTomo`.

Create a new environment and first install astra-toolbox:

```bash
conda create --name ndtomo python=3.11
conda activate ndtomo
conda install -c astra-toolbox -c nvidia astra-toolbox
```

### 2. Install nDTomo from GitHub

You can choose one of the following options to install the nDTomo library:

#### a. To install using pip:

```bash
pip install nDTomo
```

#### b. To install using Git:

```bash
pip install git+https://github.com/antonyvam/nDTomo.git
```
For development work (editable install):

```bash
git clone https://github.com/antonyvam/nDTomo.git && cd nDTomo
pip install -e .
```

#### c. For local installation after downloading the repo:

Navigate to where the `setup.py` file is located and run:

```bash
pip install --user .
```

or:

```bash
python3 setup.py install --user
```

### 3. Install PyTorch

The neural networks, as well as any GPU-based code, used in `nDTomo` require Pytorch which can be installed through pip.

For example, for Windows/Linux with CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Launching the GUI

After installing `nDTomo`, the graphical user interface can be launched directly from the terminal:

```bash
conda activate ndtomo
python -m nDTomo.gui.nDTomoGUI
```

## Diamond Light Source

As a user at the Diamond Light Source, you can install `nDTomo` by doing:

```bash
git clone https://github.com/antonyvam/nDTomo.git && cd nDTomo
module load python/3
python setup.py install --user
```

## Citation

If you use parts of the code, please cite the work using the following preprint:

*nDTomo: A Modular Python Toolkit for X-ray Chemical Imaging and Tomography*, A. Vamvakeros, E. Papoutsellis, H. Dong, R. Docherty, A.M. Beale, S.J. Cooper, S.D.M. Jacques, Digital Discovery, 2025, https://doi.org/10.1039/D5DD00252D

## References

[1] "Cycling Rate-Induced Spatially-Resolved Heterogeneities in Commercial Cylindrical Li-Ion Batteries", A. Vamvakeros, D. Matras, T.E. Ashton, A.A. Coelho, H. Dong, D. Bauer, Y. Odarchenko, S.W.T. Price, K.T. Butler, O. Gutowski, A.-C. Dippel, M. von Zimmerman, J.A. Darr, S.D.M. Jacques, A.M. Beale, Small Methods, 2100512, 2021. https://doi.org/10.1002/smtd.202100512

[2] "Emerging chemical heterogeneities in a commercial 18650 NCA Li-ion battery during early cycling revealed by synchrotron X-ray diffraction tomography", D. Matras, T.E. Ashton, H. Dong, M. Mirolo, I. Martens, J. Drnec, J.A. Darr, P.D. Quinn, S.D.M. Jacques, A.M. Beale, A. Vamvakeros, Journal of Power Sources 539, 231589, 2022, https://doi.org/10.1016/j.jpowsour.2022.231589

## Previous technical work (reverse chronological order)

[1] "Obtaining parallax-free X-ray powder diffraction computed tomography data with a self-supervised neural network", H. Dong, S.D.M. Jacques, K.T. Butler, O. Gutowski, A.-C. Dippel, M. von Zimmerman, A.M. Beale, A. Vamvakeros, npj Computational Materials 10 (1), 201, 2024, https://doi.org/10.1038/s41524-024-01389-1

[2] "SAMBA: A Trainable Segmentation Web-App with Smart Labelling", R. Docherty, I. Squires, A. Vamvakeros, S.J. Cooper, Journal of Open Source Software 9 (98), 6159, 2024, https://doi.org/10.21105/joss.06159

[3] "A scalable neural network architecture for self-supervised tomographic image reconstruction", H. Dong, S.D.M. Jacques, W. Kockelmann, S.W.T. Price, R. Emberson, D. Matras, Y. Odarchenko, V. Middelkoop, A. Giokaris, O. Gutowski, A.-C. Dippel, M. von Zimmermann, A.M. Beale, K.T. Butler, A. Vamvakeros, Digital Discovery, 2 (4), 967-980, 2023, https://doi.org/10.1039/D2DD00105E

[4] "A deep convolutional neural network for real-time full profile analysis of big powder diffraction data", H. Dong, K.T. Butler, D. Matras, S.W.T. Price, Y. Odarchenko, R. Khatry, A. Thompson, V. Middelkoop, S.D.M. Jacques, A.M. Beale, A. Vamvakeros, npj Computational Materials 7 (1), 74, 2021, https://doi.org/10.1038/s41524-021-00542-4

[5] "DLSR: a solution to the parallax artefact in X-ray diffraction computed tomography data", A. Vamvakeros, A.A. Coelho, D. Matras, H. Dong, Y. Odarchenko, S.W.T. Price, K.T. Butler, O. Gutowski, A.-C. Dippel, M. von Zimmermann, I. Martens, J. Drnec, A.M. Beale, S.D.M. Jacques, Journal of Applied Crystallography 53 (6), 1531-1541, https://doi.org/10.1107/S1600576720013576

[6] "5D operando tomographic diffraction imaging of a catalyst bed", A. Vamvakeros, S.D.M. Jacques, M. Di Michiel, D. Matras, V. Middelkoop, I.Z. Ismagilov, E.V. Matus, V.V. Kuznetsov, J. Drnec, P. Senecal, A.M. Beale, Nature communications 9 (1), 4751, 2018, https://doi.org/10.1038/s41467-018-07046-8

[7] "Interlaced X-ray diffraction computed tomography", A. Vamvakeros, S.D.M. Jacques, M. Di Michiel, P. Senecal, V. Middelkoop, R.J. Cernik and A.M. Beale, Journal of Applied Crystallography 49 (2), 485-496, 2016, https://doi.org/10.1107/S160057671600131X

[8] "Removing multiple outliers and single-crystal artefacts from X-ray diffraction computed tomography data", A. Vamvakeros, S.D.M. Jacques, M. Di Michiel, V. Middelkoop, C.K. Egan, R. J. Cernik, A. M Beale, Journal of Applied Crystallography 48 (6), 1943-1955, 2015, 
Jacques, Journal of Applied Crystallography 48 (6), 1943-1955, 2015, https://doi.org/10.1107/S1600576715020701
