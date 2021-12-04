Installation of nDTomo software
-------------------------------

Repository
^^^^^^^^^^
The nDTomo project repository is available for download from GitHub: 
https://github.com/antonyvam/nDTomo

Anaconda
^^^^^^^^

It is recommended to install first the anaconda (Python 3 version) distribution of Python as it comes with PyQt5. To install conda please see `Anaconda <https://www.anaconda.com/>`_.

Building procedure
^^^^^^^^^^^^^^^^^^

Open a terminal, navigate to the nDTomo folder where the setup.py file is locacted and run the following::

	pip install .

The GUIs can be opened using a terminal and typing::

	Integrator

	MultiTool
	
Dependencies
^^^^^^^^^^^^
nDTomo is a Python library which relies on the following packages:

* Python: version 3.8
* PyQt5
* Numpy
* Scipy
* Matplotlib
* Fabio
* Scikit-image
* h5py
* pyqtgraph
* pyFAI
* astra-toolbox


Contributors
^^^^^^^^^^^^

 * Antony Vamvakeros (Finden)
 * Simon Jacques (Finden)
 * Gavin Vaughan (ESRF)
 * Dorota Matras (Diamond Light Source)
 * Hongyang Dong (University College London)
 
Indirect contributors (ideas...)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

 * Marco di Michiel (ESRF)
 * Jérôme Kieffer (ESRF)
 
Beta testers
^^^^^^^^^^^^

 * Steve Price (Finden)
 * Donal Finegan (NREL)
 * Tom Heenan (University College London)



