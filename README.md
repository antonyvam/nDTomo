nDTomo software suite
===========
The nDTomo software suite contains GUIs for the acquisition, pre-processing and analysis of X-ray chemical tomography data

ID15A helper
-------------
A GUI to help generate .mac scripts for the ID15A beamline. These are macros that can be used during an experiment. 
The GUI helps the user design an experiment while the acquisition macros (e.g. macros for collection of X-ray tomographic data) are developed by the beamline.

Integrator
-------------
The Integrator GUI is designed to assist the user to integrate X-ray and Pair distribution function computed tomography data (XRD-CT and PDF-CT).
It can batch integrate diffraction/scattering tomographic data collected with different strategies:
* Zigzag
* Interlaced
* Continuous rotation/translation

The data integration can be performed with different processing units:
* CPU
* GPU
* ID15A data integration dedicated PC

MultiTool
-------------
The MultiTool GUI allows for the processing and analysis of chemical tomography data. For example:
1. It allows for the visualization of 3d matrices. These are typically grid maps (e.g. XRD XZ maps) or cross-sections (e.g. XRD/PDF/SAXS/XAFS/XRF-CT data) 
2. It has the ability to:
	* Centre sinogram volumes (raw chemical tomography data), 
	* Remove backround signal,
	* Normalise sinogram assuming constant total scattering intensity per projection (diffaction/scattering tomography data)
	* Reconstruct a diffaction/scattering tomography dataset
	* Reconstruct multiple diffaction/scattering tomography datasets simultaneously (batch reconstruction)
3. It can load, normalise (flat/dark images) and reconstruction of X-ray absorption computed tomography data (abs-CT)
4. It can perform a zero-order absorption correction on diffaction/scattering tomography data using an abs-CT image
5. It allows for batch single peak fitting of chemical tomography data