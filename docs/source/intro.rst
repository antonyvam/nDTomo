nDTomo software suite
=====================
The nDTomo software suite contains GUIs for the acquisition, pre-processing and analysis of X-ray chemical tomography data

Integrator
----------
The Integrator GUI is designed to assist the user to integrate X-ray scattering data. 
The data integration is performed with pyFAI. 
The integration of the diffraction data can be performed in several ways:

* Standard azimuthal integration
* Using a trimmed mean filter: the user specifies the % for trimming
* Using a new standard deviation based adaptive filter: the user specifies the number of standard deviations to be used

MultiTool
---------
The MultiTool GUI allows for the processing and analysis of chemical tomography data. For example:

1. It allows for the visualization of 3d matrices. These are typically grid maps (e.g. XRD map) or tomographic images (e.g. XRD-CT data) 
2. It has the ability to:

	* Centre sinogram volumes (raw chemical tomography data), 
	* Remove backround signal,
	* Normalise sinogram assuming constant total scattering intensity per projection (diffaction/scattering tomography data)
	* Reconstruct a diffaction/scattering tomography dataset
	* Reconstruct multiple diffaction/scattering tomography datasets simultaneously (batch reconstruction)

3. It can load, normalise (flat/dark images) and reconstruction of X-ray absorption computed tomography data (abs-CT)
4. It can perform a zero-order absorption correction on diffaction/scattering tomography data using an abs-CT image
5. It can perform batch single peak fitting of chemical imaging/tomography data
