nDTomo software suite
===========
The nDTomo software suite contains GUIs for the acquisition, pre-processing and analysis of X-ray chemical tomography data

Full documentation provided at: https://ndtomo.readthedocs.io

ID15A helper
-------------
A GUI to help generate .mac scripts for the ID15A beamline. These are macros that can be used during an experiment. 
The GUI helps the user design an experiment while the acquisition macros (e.g. macros for collection of X-ray tomographic data) are developed by the beamline.

Integrator
-------------
The Integrator GUI is designed to assist the user to integrate X-ray and Pair distribution function computed tomography data (XRD-CT and PDF-CT).
It can batch integrate diffraction/scattering tomographic data collected with different strategies [1,2]:
* Zigzag
* Interlaced
* Continuous rotation/translation

The data integration is done with pyFAI and it can be performed with different processing units [3, 4]:
* CPU
* GPU
* ID15A data integration dedicated PC

The integration of the diffraction data can be performed in several ways [5]:
* Standard azimuthal integration
* Using a trimmed mean filter: the user specifies the % for trimming
* Using a new standard deviation based adaptive filter: the user specifies the number of standard deviations to be used

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

References
-------------
1. Vamvakeros, A. et al. Interlaced X-ray diffraction computed tomography. Journal of Applied Crystallography 49, 485-496, doi:10.1107/S160057671600131X (2016).
2. Vamvakeros, A. et al. 5D operando tomographic diffraction imaging of a catalyst bed. Nature Communications 9, 4751, doi:10.1038/s41467-018-07046-8 (2018).
3. Ashiotis, G. et al. The fast azimuthal integration Python library: pyFAI. Journal of Applied Crystallography 48, 510-519, doi:10.1107/S1600576715004306 (2015).
4. Kieffer, J. et al. Real-time diffraction computed tomography data reduction. Journal of Applied Crystallography 25, 612-617, doi:10.1107/S1600577518000607 (2018).
5. Vamvakeros, A. et al. Removing multiple outliers and single-crystal artefacts from X-ray diffraction computed tomography data. Journal of Applied Crystallography 48, 1943-1955, doi:10.1107/s1600576715020701 (2015).
