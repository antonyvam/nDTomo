nDTomo GUI
==========

.. contents::
   :local:
   :depth: 2

Overview
--------

The `nDTomoGUI` provides a PyQt5-based graphical user interface (GUI) for interactive exploration and analysis of chemical imaging datasets. It is especially useful for viewing, segmenting, and analyzing hyperspectral X-ray tomography data, such as XRD-CT volumes.

The interface supports interactive ROI selection, segmentation, batch peak fitting using common profile models (Gaussian, Lorentzian, Pseudo-Voigt), and export of processed results.

Launching the GUI
-----------------

Once `nDTomo` is installed and the environment is activated, you can launch the GUI directly from the terminal:

.. code-block:: bash

    conda activate ndtomo
    nDTomoGUI

This command starts the GUI without needing to navigate to the code or run Python manually.

Tabs and Features
-----------------

The GUI consists of four main tabs, each designed to guide users through a different stage of the analysis workflow:

1. **Chemical imaging data**
   - Load `.hdf5` or `.h5` files
   - Explore spatial images and corresponding spectra
   - Change colormaps
   - Export local diffraction patterns and 2D images

2. **ROI image**
   - Define a region of interest (ROI) by selecting a channel range
   - Apply background subtraction (none, mean, linear)
   - Export the resulting ROI image

3. **ROI pattern**
   - Segment the ROI image using a threshold
   - Use the mask to extract the corresponding spectrum
   - Export the extracted ROI pattern

4. **Peak fitting**
   - Perform batch single-peak fitting using Gaussian, Lorentzian, or Pseudo-Voigt profiles
   - Monitor progress with a real-time progress bar
   - Visualize results (e.g., area, position, FWHM)
   - Export fit results as HDF5

Limitations and Notes
---------------------

- This version supports **single-peak fitting** only.
- GPU acceleration for fitting is not yet implemented.
- Only `.h5`/`.hdf5` datasets are supported for loading and saving.
- For large datasets, operations such as batch fitting may be slow — GPU-based or parallelized implementations are under consideration.

Developer Notes
---------------

The GUI is implemented as a single PyQt5-based script with the main class:

.. autoclass:: nDTomo.gui.nDTomoGUI.nDTomoGUI
    :members:
    :undoc-members:
    :show-inheritance:

Fitting is executed in a separate thread using:

.. autoclass:: nDTomo.gui.nDTomoGUI.FitData
    :members:
    :undoc-members:

The GUI entry point is declared in `setup.py` using `gui_scripts`, which creates the `nDTomoGUI` command on install.
