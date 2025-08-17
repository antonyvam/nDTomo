# -*- coding: utf-8 -*-
"""
Hyperspectral Imaging Explorers and GUI Tools

This module provides several interactive classes for exploring hyperspectral or volumetric image data, particularly for chemical imaging datasets.
It includes mouse-interactive tools for visualizing spectra, image slices, and intensity profiles. F

Classes:
    - HyperSliceExplorer: Explore hyperspectral imaging data
    - ImageSpectrumGUI: Explore hyperspectral imaging data
    - InteractiveProfileExtraction: Extract 1D intensity profiles along a line in a 2D image
    - InteractiveHyperProfileExtraction: Extract 1D and spectral profiles along a line in a hyperspectral volume
    - ImageSpectrumFitGUI: Compare raw and fitted spectra from voxel data interactively
    - chemimexplorer: A version of ImageSpectrumGUI which is used as embedded in jupyter notebooks

@author: Antony Vamvakeros
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates
from matplotlib.lines import Line2D

class HyperSliceExplorer():
        
    """
    Interactive explorer for hyperspectral or volumetric imaging data.

    Allows the user to click and move the mouse over a mean image view 
    to inspect voxel-level spectra and update an associated spectrum plot. 
    Users can visualize individual channel images and view spectral changes dynamically.
    Particularly useful for visual inspection of 3D chemical imaging datasets.
    """
    
    def __init__(self, data, xaxis=None, xaxislabel='Channels'):


        """
        Initialize the HyperSliceExplorer object.

        Sets up the internal state and prepares data structures for 
        interactive visualisation of 3D hyperspectral (or chemical imaging) data.

        Parameters
        ----------
        data : np.ndarray
            A 3D array of shape (rows, cols, channels) representing the volumetric dataset
            where each (x, y) pixel contains a spectrum.
        xaxis : np.ndarray, optional
            A 1D array representing the spectral axis (e.g., 2θ, q, or energy).
            If None, defaults to a linear range [0, N) where N is the number of channels.
        xaxislabel : str, optional
            A string label for the x-axis used in spectrum plots. Defaults to 'Channels'.

        Attributes
        ----------
        data : np.ndarray
            The input hyperspectral dataset.
        xaxis : np.ndarray
            Spectral axis used for plotting spectra.
        xaxislabel : str
            Label shown on spectrum plots.
        cmap : str
            Colormap used for image rendering.
        imoi : np.ndarray
            Mean image across all channels.
        mdp : np.ndarray
            Mean spectrum across all spatial positions.
        sdp : np.ndarray
            Summed spectrum across all spatial positions.
        dproi : np.ndarray
            Most recently selected spectrum (by click).
        selectedChannels : list or int
            Channel(s) currently selected for image display.
        selectedVoxels : list
            List of selected voxel positions.
        map_fig, map_axes, plot_fig, plot_axes : matplotlib components
            Handles to matplotlib figures and axes for image and spectral views.
        """
        
        self.data = data
        if xaxis is None:
            self.xaxis = np.arange(0,data.shape[2])
        else:
            self.xaxis = xaxis
            
        self.xaxislabel = xaxislabel
        
        self.mapper = []
        self.map_fig = []
        self.map_axes = []
        self.map_data = []
        self.mapHoldState= 0;
        self.plotter = []
        self.plot_fig = []
        self.plot_axes = []
        self.currentCurve = 0;
        self.plot = []
        self.plotHoldState= 0;
        self.selectedChannels = []
        self.selectedVoxels = np.empty(0,dtype=object)
        self.meanSpectrum = np.mean(np.mean(data, axis = 0), axis = 0)
        self.cmap = 'jet'

        self.imoi = np.mean(self.data,axis=2)
        self.mdp = np.mean(np.mean(self.data,axis=1),axis=0)
        self.sdp = np.sum(np.sum(self.data,axis=1),axis=0)
        self.dproi = self.mdp

    def explore(self, cmap = 'jet'):

        """
        Launch interactive exploration of hyperspectral data.

        This method creates two interactive Matplotlib figures:
        - A mean image view that supports mouse hovering and clicking to inspect voxel-level spectra.
        - A spectrum plot that updates dynamically based on cursor position over the image or spectrum.

        Users can click:
        - On the image (left-click) to fix and label a specific voxel's spectrum.
        - On the spectrum (right-click) to re-enable hover-based channel selection.

        Parameters
        ----------
        cmap : str, optional
            Colormap used for image display. Default is 'jet'.

        Notes
        -----
        - A vertical line (vCurve) indicates the current spectral channel under inspection.
        - Uses Matplotlib event callbacks to manage interactive behavior.
        - Updates `self.dproi`, `self.histogramCurve`, and `self.activeCurve` for spectrum display.
        """
                
        self.cmap = cmap
        
        self.mapper = plt;
        self.map_fig = self.mapper.figure()
        self.map_axes = self.map_fig.add_subplot(111)        
        self.map_data = self.mapper.imshow(self.imoi, cmap=self.cmap)
        title = 'Mean image'
        self.mapper.title(title, fontstyle='italic')
        self.map_fig.canvas.mpl_connect('button_press_event', self.onMapClick)
        self.map_fig.canvas.mpl_connect('motion_notify_event', self.onMapMoveEvent)        
        
        self.cb = self.map_fig.colorbar(self.map_data)
        
        self.mapper.show()
        self.mapper.draw()  
        
        
        self.plotter = plt
        self.plot_fig = self.plotter.figure()
        self.plot_axes = self.plot_fig.add_subplot(111)        
        self.plotter.legend()
        self.plotter.title("Mean spectrum", fontstyle='italic')
        
        self.plot_fig.canvas.mpl_connect('motion_notify_event', self.onPlotClick)       
        self.plot_cid = self.plot_fig.canvas.mpl_connect('motion_notify_event', self.onPlotMoveEvent)   
        self.plotter.show()        
        self.plotter.draw()        

        self.histogramCurve = self.plotter.plot(self.xaxis, self.mdp, label='Mean spectrum', color="b")
        self.activeCurve = self.plotter.plot(self.xaxis, self.mdp, label='Mean spectrum', color="r")
        
        ########
        self.vCurve = self.plotter.axvline(x=0, color="k")
        #######
                
        
    def onMapClick(self, event): # SDMJ version
    
        """
        Handle mouse click events on the image panel.

        Left-click:
            - Captures the coordinates of the clicked voxel (row, col).
            - Extracts and displays the corresponding spectrum in the spectral plot.
            - Updates the label to show voxel location.

        Right-click:
            - Hides the fixed spectrum curve.
            - Rescales the y-axis to fit the last selected voxel spectrum.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The mouse click event containing button info and coordinates.

        Notes
        -----
        - Left-click outside the axes resets the spectrum plot to the mean spectrum.
        - This function updates `self.dproi`, `self.histogramCurve`, and plot limits.
        """
            
        if event.button == 1:
            if event.inaxes:
                self.col = int(event.xdata.round())
                self.row = int(event.ydata.round())
                
                self.dproi = self.data[self.row,self.col,:]
                
                self.histogramCurve[0].set_data(self.xaxis, self.dproi) 
                self.histogramCurve[0].set_label(str(self.row)+','+str(self.col))
                self.histogramCurve[0].set_visible(True)          
                self.plotter.legend()
                
                self.plotter.show()        
                self.plotter.draw()

            else:
                self.selectedVoxels = np.empty(0,dtype=object)
                self.plot_axes.clear() # not fast
                self.plotter.plot(self.xaxis, self.mdp, label='Mean spectrum')
                self.plotter.legend()                
                self.plotter.draw() 

        if event.button == 3:
            if event.inaxes:
                self.histogramCurve[0].set_visible(False)          
                
                self.plot_axes.set_xlim(self.xaxis[0],self.xaxis[-1])
                self.plot_axes.set_ylim(0,np.max(self.dproi))
                self.plotter.show()        
                self.plotter.draw()

    def onMapMoveEvent(self, event): 
        
        """
        Handle mouse movement over the image panel.

        When the cursor is over the image axes:
            - Displays the spectrum corresponding to the hovered voxel (row, col).
            - Updates the red "activeCurve" line in the spectrum plot.
            - Updates the legend to show voxel coordinates.
            - Dynamically adjusts the y-axis limits based on the hovered spectrum.

        When the cursor leaves the image axes:
            - Hides the red "activeCurve".
            - Resets y-axis to fit the last clicked spectrum (`self.dproi`).

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The mouse motion event with axis and coordinate info.

        Notes
        -----
        This function provides real-time spectral feedback for hyperspectral data exploration.
        """
        
        if event.inaxes:
            
            col = int(event.xdata.round())
            row = int(event.ydata.round())
            
            dproi = self.data[row,col,:]            
            
            self.activeCurve[0].set_data(self.xaxis, dproi) 
            self.activeCurve[0].set_label(str(row)+','+str(col))
            self.activeCurve[0].set_visible(True)
            if np.max(dproi)>0:
                self.plot_axes.set_ylim(0,np.max(dproi))

        else:
            self.activeCurve[0].set_visible(False)
            self.activeCurve[0].set_label('')
            self.plot_axes.set_ylim(0,np.max(self.dproi))
            
        self.plotter.legend()
        self.plotter.draw()
    
    def onPlotMoveEvent(self, event):
        
        """
        Handle mouse movement over the spectral plot panel.

        Updates the left-hand image panel in real-time to show the spatial image 
        corresponding to the spectral channel under the cursor.

        Actions performed:
        - Determines the nearest spectral channel (index `nx`) based on x-coordinate.
        - Updates the image (`imoi`) to show the spatial slice at `nx`.
        - Updates the title with channel and axis label (e.g., energy, 2θ).
        - Replaces the colorbar and redraws the figure.
        - Moves the vertical guide line (`vCurve`) on the spectrum plot to follow the cursor.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The mouse motion event with axis and coordinate information.
        """
                
        if event.inaxes:
                        
            nx = np.argmin(np.abs(self.xaxis-event.xdata))
            
            if nx<0:
                nx = 0
            elif nx>len(self.xaxis):
                nx = len(self.xaxis)-1

            self.selectedChannels = nx
            self.map_axes.clear() # not fast
            self.imoi = self.data[:,:,nx]
            self.map_data = self.map_axes.imshow(self.imoi, cmap = self.cmap)
            title = "Channel = %d; %s = %.3f" % (nx, self.xaxislabel, self.xaxis[nx])

            try:
                self.cb.remove()
            except:
                pass
            
            self.cb = self.map_fig.colorbar(self.map_data)
            self.map_axes.set_title(title)
            self.mapper.draw_all()             
            self.vCurve.set_xdata(event.xdata) 
            self.plotter.draw()
     
    def onPlotClick(self, event):

        """
        Handle mouse click events on the spectral plot.

        Left-click (button 1) disables channel hover updates by disconnecting 
        the motion event that controls image updates.

        Right-click (button 3) re-enables channel hover updates by reconnecting
        the motion event handler to update the image view based on cursor position.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The mouse click event.
        """
        if event.button == 1:
            self.plot_fig.canvas.mpl_disconnect(self.plot_cid)   

        elif event.button == 3:
            self.plot_cid = self.plot_fig.canvas.mpl_connect('motion_notify_event', self.onPlotMoveEvent)
            self.plotter.show()  
            self.plotter.draw()   
            
    def update(self):
        
        """
        Refresh the left-hand image display based on the currently selected channels.

        If no specific channels are selected, displays the mean image across all channels.
        If a single channel is selected, displays that channel.
        If multiple channels are selected, displays the mean across those channels.

        Also updates the image title and redraws the figure with the current colormap.
        """
                
        self.map_axes.clear() # not fast
        # this bit is messy
        
        self.imoi = np.mean(self.data, axis = 2)
        if (not self.selectedChannels):
            self.map_axes.imshow(self.imoi ,cmap=self.cmap)
            title = 'Mean image'
        else:
            if self.selectedChannels.size == 1:
                self.map_axes.imshow(self.data[:,:,self.selectedChannels],cmap=self.cmap)
                title = "Channel = %d; %s = %.3f" % (self.nx, self.xaxislabel, self.xaxis[self.nx])
            if self.selectedChannels.size > 1:
                self.map_axes.imshow(np.mean(self.data[:,:,self.selectedChannels],2),cmap=self.cmap)
                title = self.name+' '+'mean of selected channels'
        self.map_axes.set_title(title)
        self.mapper.show()
        self.mapper.draw_all() 


class ImageSpectrumGUI:
    
    """
    Interactive viewer for 2D hyperspectral images (rows x cols x channels).

    The left pane shows an image for a chosen spectral band; the right pane
    shows the spectrum at the current mouse position. Hovering over the image
    updates the spectrum to the voxel (x, y) under the cursor. Hovering over
    the spectrum updates the image to the spectral band under the cursor
    (i.e., quick band scrubbing).

    Right-click toggles “live” updates independently for each pane so you can
    freeze either the image or the spectrum while inspecting the other.

    If an optional boolean `mask` is provided, it is overlaid in color on the
    image view to help assess segmentation quality (uses a semi-transparent
    overlay). This is useful for QA of thresholding/segmentation against the
    raw grayscale signal.

    Notes
    -----
    - Expected input shape is (rows, cols, channels). Displayed images are
      normalized for visualization only; underlying data are unchanged.
    - The overlay `mask` should match `volume.shape[:2]` or `volume.shape`
      (if channel-wise masks); when 3D it uses the slice `mask[:, :, band]`.
    - Designed for quick exploratory analysis without precomputations; large
      volumes are fine as only the needed slices/vectors are drawn on demand.
    """
    
    def __init__(self, volume, cmap='jet', mask=None):
        
        """
        Initialize the interactive viewer.

        Parameters
        ----------
        volume : ndarray
            Hyperspectral data with shape (rows, cols, channels). The left
            panel starts from the mean image over channels; the right panel
            starts from the mean spectrum over (rows, cols). Moving the mouse
            updates the other panel in real time.
        cmap : str, optional
            Matplotlib colormap for the image view (default: 'jet'). Ignored
            when a color overlay is rendered by `mask` for a given band.
        mask : ndarray of bool, optional
            Segmentation mask to overlay on the image view for visual QA.
            shape == (rows, cols, channels): the band-specific mask
            `mask[:, :, band]` is shown when that band is displayed.

        Interaction
        -----------
        - Hover over image: updates spectrum at cursor (x, y).
        - Hover over spectrum: updates image to nearest spectral band index.
        - Right-click on a panel: toggles live updates for that panel on/off.

        """

        self.volume = volume
        if mask is not None:
            self.mask = mask
        else:
            self.mask = None

        # Create main figure and subplots
        self.fig, (self.ax_image, self.ax_spectrum) = plt.subplots(1, 2, figsize=(10, 5))

        # Initialize with mean image and mean spectrum
        mean_image = np.mean(volume, axis=2)
        mean_spectrum = np.mean(volume, axis=(0, 1))

        # Plot the mean image and mean spectrum
        self.image = self.ax_image.imshow(mean_image.T, cmap=cmap)
        self.spectrum, = self.ax_spectrum.plot(mean_spectrum, color='b')

        # Set titles for image and spectrum
        self.ax_image.set_title('Image')
        self.ax_spectrum.set_title('Spectrum')

        # Connect mouse hover events
        self.fig.canvas.mpl_connect('motion_notify_event', self.update_plots)
        self.fig.canvas.mpl_connect('button_press_event', self.toggle_real_time)

        # Initialize real-time update flags
        self.image_real_time_update = True
        self.spectrum_real_time_update = True

    def update_plots(self, event):
        """
        Handle mouse movement over the figure to update image or spectrum.

        If hovering over the image panel and image updates are enabled, 
        it displays the spectrum of the voxel under the cursor.

        If hovering over the spectrum panel and spectrum updates are enabled, 
        it displays the image corresponding to the spectral bin under the cursor.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Event object containing cursor position and axis context.
        """
        if event.inaxes == self.ax_image:
            if not self.image_real_time_update:
                return
            x, y = int(event.xdata), int(event.ydata)

            # Check if the mouse position is within the image dimensions
            image_width, image_height, _ = self.volume.shape
            if x >= 0 and x < image_width and y >= 0 and y < image_height:
                # Get the spectrum from the volume
                spectrum = self.volume[x, y, :]

                # Update the spectrum plot
                self.spectrum.set_ydata(spectrum)
                self.ax_spectrum.relim()
                self.ax_spectrum.autoscale_view()

                # Set title with coordinates
                self.ax_spectrum.set_title(f'Spectrum (x={x}, y={y})')

        elif event.inaxes == self.ax_spectrum:
            if not self.spectrum_real_time_update:
                return
            index = int(event.xdata)

            # Get the image from the volume
            image = self.volume[:, :, index]

            if self.mask is not None:
                image = label2rgb(self.mask[:, :, index], image=image,
                                  bg_label=0, alpha=0.35,
                                  colors=[(1, 0, 0)])

            # Update the image display
            if self.mask is not None:
                self.image.set_data(image.transpose(1,0,2))
            else:
                self.image.set_data(image.T)
            self.ax_image.relim()
            self.ax_image.autoscale_view()

            # Set title with bin
            self.ax_image.set_title(f'Image (Bin={index})')

            # Set color limits based on the current image
            self.image.set_clim(np.min(image), np.max(image))

        self.fig.canvas.draw()

    def toggle_real_time(self, event):
        """
        Toggle real-time interactivity for image or spectrum on right-click.

        Disables/enables automatic updates depending on the axis clicked.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Mouse button press event. Right-click toggles interactivity.
        """
        if event.button == 3:
            if event.inaxes == self.ax_image:
                self.image_real_time_update = not self.image_real_time_update
            elif event.inaxes == self.ax_spectrum:
                self.spectrum_real_time_update = not self.spectrum_real_time_update



class InteractiveProfileExtraction:
    
    """
    Interactive tool for extracting 1D intensity profiles from 2D grayscale images.

    Enables users to draw a line on an image, then extracts and visualizes the
    intensity values along that path. Useful for analyzing gradients, edges,
    and intensity distributions across features in microscopy or tomography images.

    Features
    --------
    - Interactive line drawing on the image canvas.
    - Real-time line preview during mouse motion.
    - Profile extracted using Bresenham's algorithm and interpolated coordinates.
    - Three-panel display: image, path overlay, and profile plot.
    """
        
    def __init__(self, image):
        
        """
        Initialize the InteractiveProfileExtraction tool.

        Parameters
        ----------
        image : ndarray
            2D grayscale image from which intensity profiles will be interactively extracted.
            The user defines a line on the image, and the corresponding intensity values are plotted.
        """
                
        self.image = image
        self.line = None
        self.profile = None

        self.fig, self.axes = plt.subplots(1, 3, figsize=(12, 4))
        self.axes[0].imshow(self.image, cmap='gray')
        self.axes[0].set_title("Interactive Profile Extraction")
        self.axes[1].set_title("Intensity Profile")
        self.axes[1].set_xlabel("Distance (pixels)")
        self.axes[1].set_ylabel("Intensity")
        self.axes[2].set_xlabel('Position along the line')
        self.axes[2].set_ylabel('Intensity')
        self.axes[2].set_title('Intensity Profile')

        self.line_completed = False

        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        
        """
        Handle mouse click events for defining the profile line.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Mouse event used to place or complete the line.
        """
                
        if event.button == 1 and event.inaxes == self.axes[0]:
            if self.line is None:
                self.profile = [event.ydata, event.xdata, event.ydata, event.xdata]
                self.line = Line2D([self.profile[1], self.profile[3]], [self.profile[0], self.profile[2]], color='red')
                self.axes[0].add_line(self.line)
                self.fig.canvas.draw()
            else:
                self.profile[2] = event.ydata
                self.profile[3] = event.xdata
                self.line.set_xdata([self.profile[1], self.profile[3]])
                self.line.set_ydata([self.profile[0], self.profile[2]])
                self.fig.canvas.draw()
                if not self.line_completed:
                    self.line_completed = True
                    self.extract_intensity_profile()
                    self.show_intensity_profile()

    def on_motion(self, event):
        
        """
        Update the profile line dynamically during mouse movement.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Mouse movement event over the image axis.
        """        
        if self.line is not None and event.inaxes == self.axes[0] and not self.line_completed:
            self.profile[2] = event.ydata
            self.profile[3] = event.xdata
            self.line.set_xdata([self.profile[1], self.profile[3]])
            self.line.set_ydata([self.profile[0], self.profile[2]])
            self.fig.canvas.draw()

    def extract_intensity_profile(self):
        
        """
        Extract intensity values from the image along the defined line.

        Uses Bresenham’s algorithm to identify discrete pixel coordinates along
        the user-drawn line. Passes results to the plotting routine.
        """
                
        y0, x0, y1, x1 = map(int, self.profile)
        yy, xx = bresenham(y0, x0, y1, x1)
        intensity_profile = self.image[yy, xx]
        self.plot_intensity_profile(xx, yy, intensity_profile)

    def plot_intensity_profile(self, xx, yy, intensity_profile):
        """
        Display the image with the intensity profile line overlayed.

        Parameters
        ----------
        xx : array-like
            X (column) coordinates of the profile path.
        yy : array-like
            Y (row) coordinates of the profile path.
        intensity_profile : array-like
            Intensity values along the drawn line.
        """        
        self.axes[1].clear()
        self.axes[1].imshow(self.image, cmap='gray')
        self.axes[1].plot(yy, xx, 'r-')
        self.axes[1].set_title("Intensity Profile")
        self.axes[1].set_xlabel("Distance (pixels)")
        self.axes[1].set_ylabel("Intensity")

    def show_intensity_profile(self):

        """
        Interpolate and plot the 1D intensity profile along the drawn line.

        Uses scipy.ndimage.map_coordinates for subpixel interpolation.
        """        
        self.y_coords, self.x_coords = self.get_line_coordinates(self.profile[0], self.profile[1],
                                                       self.profile[2], self.profile[3])
        self.intensity_values = map_coordinates(self.image, np.vstack((self.y_coords,self.x_coords)))
        self.axes[2].clear()
        self.axes[2].plot(self.intensity_values)
        self.axes[2].set_xlabel('Position along the line')
        self.axes[2].set_ylabel('Intensity')
        self.axes[2].set_title('Intensity Profile')
        self.fig.canvas.draw()

    def clear_profile(self):
        """
        Clear the currently drawn line and reset the profile extraction state.

        Removes the line overlay from the image, resets internal flags,
        and clears stored profile information.
        """        
        if self.line is not None:
            self.line.remove()
            self.line = None
            self.line_completed = False
            self.fig.canvas.draw()
            self.profile = None

    def get_line_coordinates(self, y1, x1, y2, x2):
        """
        Generate integer coordinates along a straight line between two points using linear interpolation.

        Parameters
        ----------
        y1, x1 : float
            Row and column of the starting point.
        y2, x2 : float
            Row and column of the ending point.

        Returns
        -------
        y : ndarray
            Row indices along the line.
        x : ndarray
            Column indices along the line.
        """        
        length = int(np.hypot(y2 - y1, x2 - x1))
        y, x = np.linspace(y1, y2, length), np.linspace(x1, x2, length)
        return y.astype(int), x.astype(int)

def bresenham(y0, x0, y1, x1):
    """
    Generate pixel coordinates of a line between two points using Bresenham's algorithm.

    This discrete algorithm is efficient for rasterizing lines in 2D arrays. It ensures that
    only valid integer pixel coordinates are returned, which is useful for fast profile
    extraction from images.

    Parameters
    ----------
    y0, x0 : int
        Row and column of the starting point.
    y1, x1 : int
        Row and column of the ending point.

    Returns
    -------
    coords : ndarray, shape (2, N)
        Array of (x, y) coordinates along the line as two 1D arrays stacked vertically.
        These can be used directly to index into 2D arrays: e.g., image[coords[1], coords[0]]
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    steep = dy > dx
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    dx = x1 - x0
    dy = y1 - y0
    error = int(dx / 2.0)
    ystep = 1 if y0 < y1 else -1
    y = y0
    points = []
    for x in range(x0, x1 + 1):
        coord = (y, x) if steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx
    return np.array(points).T


class InteractiveHyperProfileExtraction:
    
    """
    Interactive profile extraction tool for 2D hyperspectral imaging data.

    Enables users to draw a line on a 2D projection of the volume 
    and extract both spatial intensity profiles and corresponding 
    spectral information along the line. Useful for linking spatial and 
    spectral variations in hyperspectral datasets.
    """
        
    def __init__(self, vol):
        
        """
        Initialize the InteractiveHyperProfileExtraction tool.

        Parameters
        ----------
        vol : ndarray
            3D hyperspectral or volumetric image data with shape (rows, cols, channels).
            The tool enables drawing a line on the projection to extract both spatial and spectral profiles.
        """
                
        self.vol = vol
        self.mean_image = np.sum(self.vol, axis=2)
        self.line = None
        self.profile = None

        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 12))
        self.axes[0, 0].imshow(self.mean_image, cmap='gray')
        self.axes[0, 0].set_title("Interactive Profile Extraction")
        self.axes[1, 0].set_title("Intensity Profile")
        self.axes[1, 0].set_xlabel("Distance (pixels)")
        self.axes[1, 0].set_ylabel("Intensity")
        self.axes[0, 1].set_xlabel('Position along the line')
        self.axes[0, 1].set_ylabel('Intensity')
        self.axes[0, 1].set_title('Intensity Profile')
        self.axes[1, 1].set_xlabel('Position along the line')
        self.axes[1, 1].set_ylabel('Intensity')
        self.axes[1, 1].set_title('Spectral Profile')

        self.line_completed = False

        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        
        """
        Handle mouse click to initiate or complete a line selection.

        Left click defines the start and end points of the line.
        Once the line is completed, intensity and spectral profiles are computed.
        
        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Mouse event captured from the figure canvas.
        """        
        if event.button == 1 and event.inaxes is not None and event.inaxes in [self.axes[0, 0]]:
            if self.line is None:
                self.profile = [event.ydata, event.xdata, event.ydata, event.xdata]
                self.line = Line2D([self.profile[1], self.profile[3]], [self.profile[0], self.profile[2]], color='red')
                self.axes[0, 0].add_line(self.line)
                self.fig.canvas.draw()
            else:
                self.profile[2] = event.ydata
                self.profile[3] = event.xdata
                self.line.set_xdata([self.profile[1], self.profile[3]])
                self.line.set_ydata([self.profile[0], self.profile[2]])
                self.fig.canvas.draw()
                if not self.line_completed:
                    self.line_completed = True
                    self.extract_intensity_profile()
                    self.show_intensity_profile()
                    self.show_spectral_profile()

    def on_motion(self, event):

        """
        Update the end point of the line dynamically during mouse movement.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Mouse event captured during movement across the figure.
        """
                
        if self.line is not None and event.inaxes is not None and event.inaxes in [self.axes[0, 0]]:
            if not self.line_completed:
                self.profile[2] = event.ydata
                self.profile[3] = event.xdata
                self.line.set_xdata([self.profile[1], self.profile[3]])
                self.line.set_ydata([self.profile[0], self.profile[2]])
                self.fig.canvas.draw()

    def extract_intensity_profile(self):
        """
        Extract the spatial intensity profile from the 2D mean image 
        along the selected line using the Bresenham algorithm.
        """        
        y0, x0, y1, x1 = map(int, self.profile)
        yy, xx = bresenham(y0, x0, y1, x1)
        intensity_profile = self.mean_image[yy, xx]
        self.plot_intensity_profile(xx, yy, intensity_profile)

    def plot_intensity_profile(self, xx, yy, intensity_profile):
        """
        Overlay the extracted profile line on the image and replot it.

        Parameters
        ----------
        xx, yy : ndarray
            Arrays of x and y pixel coordinates along the profile.
        intensity_profile : ndarray
            Intensity values along the drawn line.
        """        
        self.axes[1,0].clear()
        self.axes[1,0].imshow(self.mean_image, cmap='gray')
        self.axes[1,0].plot(yy, xx, 'r-')
        self.axes[1,0].set_title("Intensity Profile")
        self.axes[1,0].set_xlabel("Distance (pixels)")
        self.axes[1,0].set_ylabel("Intensity")

    def show_intensity_profile(self):
        """
        Plot the interpolated 1D intensity profile along the drawn line.

        Uses linear interpolation via `scipy.ndimage.map_coordinates` to extract
        smooth values along subpixel positions.
        """        
        self.y_coords, self.x_coords = self.get_line_coordinates(self.profile[0], self.profile[1],
                                                       self.profile[2], self.profile[3])
        self.intensity_values = map_coordinates(self.mean_image, np.vstack((self.y_coords,self.x_coords)))
        self.axes[0,1].clear()
        self.axes[0,1].plot(self.intensity_values)
        self.axes[0,1].set_xlabel('Position along the line')
        self.axes[0,1].set_ylabel('Intensity')
        self.axes[0,1].set_title('Intensity Profile')
        self.fig.canvas.draw()

    def show_spectral_profile(self):
        """
        Extract and plot spectra from each pixel along the drawn line.

        Displays stacked spectra with slight vertical offsets to highlight
        spectral variation along the spatial profile.
        """
        self.y_coords, self.x_coords = self.get_line_coordinates(self.profile[0], self.profile[1],
                                                       self.profile[2], self.profile[3])
    
        sorted_indices = np.argsort(self.x_coords)
        self.sorted_x_coords = self.x_coords[sorted_indices]
        self.sorted_y_coords = self.y_coords[sorted_indices]
    
        self.spectra = np.zeros((len(self.sorted_x_coords),self.vol.shape[2]), dtype = 'float32')
        
        self.axes[1, 1].clear()
    
        ii = 0
        for y, x in zip(self.sorted_y_coords, self.sorted_x_coords):
            spectrum = self.vol[y, x, :]
            self.spectra[ii] = spectrum
            self.axes[1, 1].plot(spectrum + ii * 0.1)
            ii = ii + 1


        self.axes[1,1].set_xlabel('Position along the line')
        self.axes[1,1].set_ylabel('Intensity')
        self.axes[1,1].set_title('Spectral Profile')
        self.fig.canvas.draw()
        
    def clear_profile(self):
        """
        Clear the currently drawn line and reset the tool state.
        """        
        if self.line is not None:
            self.line.remove()
            self.line = None
            self.line_completed = False
            self.fig.canvas.draw()
            self.profile = None

    def get_line_coordinates(self, y1, x1, y2, x2):
        """
        Generate pixel coordinates along a line between two points using linear interpolation.

        Parameters
        ----------
        y1, x1 : float
            Start point of the line.
        y2, x2 : float
            End point of the line.

        Returns
        -------
        y : ndarray
            Array of row indices.
        x : ndarray
            Array of column indices.
        """        
        length = int(np.hypot(y2 - y1, x2 - x1))
        y, x = np.linspace(y1, y2, length), np.linspace(x1, x2, length)
        return y.astype(int), x.astype(int)


class ImageSpectrumFitGUI:
    
    """
    GUI for visual comparison between raw and fitted hyperspectral data.

    Displays original and fitted spectra from a 3D volume on a voxel-by-voxel basis.
    Supports dynamic inspection by mouse interaction. Useful for verifying 
    the quality of spectral fitting or peak decomposition across a sample.
    """
        
    def __init__(self, volume, volfit):
        
        """
        Initialize the ImageSpectrumFitGUI for raw vs. fitted spectrum comparison.

        Parameters
        ----------
        volume : ndarray
            3D array of original hyperspectral data with shape (rows, cols, channels).
        volfit : ndarray
            3D array of fitted spectral data with the same shape as `volume`.
            The GUI displays both raw and fitted spectra for user-selected voxels.
        """
        
        self.volume = volume
        self.volfit = volfit

        # Create main figure and subplots
        self.fig, (self.ax_image, self.ax_spectrum) = plt.subplots(1, 2, figsize=(10, 5))

        # Initialize with mean image and mean spectrum
        mean_image = np.mean(volume, axis=2)
        mean_spectrum = np.mean(volume, axis=(0, 1))
        mean_spectrum_fit = np.mean(volfit, axis=(0, 1))

        # Plot the mean image and mean spectrum
        self.image = self.ax_image.imshow(mean_image.T, cmap='gray')
        self.spectrum, = self.ax_spectrum.plot(mean_spectrum, 'bo', label='Original Spectrum')
        self.spectrum_fit, = self.ax_spectrum.plot(mean_spectrum_fit, 'r-', label='Fit Spectrum')

        # Set titles for image and spectrum
        self.ax_image.set_title('Image')
        self.ax_spectrum.set_title('Spectrum')

        # Add legend to the spectrum plot
        self.ax_spectrum.legend()

        # Connect mouse hover events
        self.fig.canvas.mpl_connect('motion_notify_event', self.update_plots)
        self.fig.canvas.mpl_connect('button_press_event', self.toggle_real_time)

        # Initialize real-time update flags
        self.image_real_time_update = True
        self.spectrum_real_time_update = True

    def update_plots(self, event):
        """
        Update the displayed spectrum and image based on mouse hover.

        When the cursor hovers over the image panel, updates the spectrum 
        panel to show the corresponding raw and fitted spectra at that voxel.

        When hovering over the spectrum panel, switches the image display 
        to show the spatial distribution at the selected spectral bin.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The mouse event object containing position and axes info.
        """
        if event.inaxes == self.ax_image:
            if not self.image_real_time_update:
                return
            x, y = int(event.xdata), int(event.ydata)

            # Check if the mouse position is within the image dimensions
            image_width, image_height, _ = self.volume.shape
            if x >= 0 and x < image_width and y >= 0 and y < image_height:
                # Get the spectrum from the volumes
                spectrum = self.volume[x, y, :]
                spectrum_fit = self.volfit[x, y, :]

                # Update the spectrum plot with both original and fit
                self.spectrum.set_ydata(spectrum)
                self.spectrum_fit.set_ydata(spectrum_fit)
                self.ax_spectrum.relim()
                self.ax_spectrum.autoscale_view()

                # Set title with coordinates
                self.ax_spectrum.set_title(f'Spectrum (x={x}, y={y})')

        elif event.inaxes == self.ax_spectrum:
            if not self.spectrum_real_time_update:
                return
            index = int(event.xdata)

            # Get the image from the volume
            image = self.volume[:, :, index]

            # Update the image display
            self.image.set_data(image.T)
            self.ax_image.relim()
            self.ax_image.autoscale_view()

            # Set title with bin
            self.ax_image.set_title(f'Image (Bin={index})')

            # Set color limits based on the current image
            self.image.set_clim(np.min(image), np.max(image))

        self.fig.canvas.draw()

    def toggle_real_time(self, event):
        """
        Toggle real-time updates for spectrum and image panels on right-click.

        Right-clicking over the image panel enables/disables updates triggered by 
        mouse motion over it. Same applies for the spectrum panel.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The mouse button press event.
        """
        if event.button == 3:
            if event.inaxes == self.ax_image:
                self.image_real_time_update = not self.image_real_time_update
            elif event.inaxes == self.ax_spectrum:
                self.spectrum_real_time_update = not self.spectrum_real_time_update
                
class chemimexplorer:
    
    """
    Lightweight GUI for Jupyter-based exploration of chemical imaging volumes.

    Provides linked spatial and spectral views for both raw and optionally 
    fitted datasets. Supports mouse-based interaction to inspect individual 
    spectra. Optimized for use within notebook environments.
    """
        
    def __init__(self, volume, fitted=None):
        
        """
        Initialize the chemimexplorer widget for use in notebooks.

        Parameters
        ----------
        volume : ndarray
            2D hyperspectral imaging dataset with shape (rows, cols, channels).
        fitted : ndarray, optional
            Optional 3D array of fitted spectral data with the same shape as `volume`.
            If provided, fitted spectra are displayed alongside raw spectra.
        """
        
        self.volume = volume
        self.fitted = fitted

        # Create main figure and subplots
        self.fig, (self.ax_image, self.ax_spectrum) = plt.subplots(1, 2, figsize=(10, 5))

        # Initialize with mean image and mean spectrum
        mean_image = np.mean(volume, axis=2)
        mean_spectrum = np.mean(volume, axis=(0, 1))
        if self.fitted is not None:
            mean_spectrum_fitted = np.mean(self.fitted, axis=(0, 1))

        # Plot the mean image and mean spectrum
        self.image = self.ax_image.imshow(mean_image.T, cmap='gray')
        if self.fitted is not None:
            self.spectrum, = self.ax_spectrum.plot(mean_spectrum, 'b+')
            self.spectrum_fitted, = self.ax_spectrum.plot(mean_spectrum_fitted, color='r')
        else:
            self.spectrum, = self.ax_spectrum.plot(mean_spectrum, color='b')

        # Set titles for image and spectrum
        self.ax_image.set_title('Image')
        self.ax_spectrum.set_title('Histogram')

        # Connect mouse hover events
        self.fig.canvas.mpl_connect('motion_notify_event', self.update_plots)
        self.fig.canvas.mpl_connect('button_press_event', self.toggle_real_time)

        # Initialize real-time update flags
        self.image_real_time_update = True
        self.spectrum_real_time_update = True

    def update_plots(self, event):
        """
        Update the displayed spectrum and image based on mouse hover.

        - Hovering over the image updates the spectrum view for the selected voxel.
        - Hovering over the spectrum panel updates the image view for the selected channel.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The mouse event object containing position and axes info.
        """
        if event.inaxes == self.ax_image:
            if not self.image_real_time_update:
                return
            x, y = int(event.xdata), int(event.ydata)

            # Check if the mouse position is within the image dimensions
            image_width, image_height, _ = self.volume.shape
            if x >= 0 and x < image_width and y >= 0 and y < image_height:
                # Get the spectrum from the volume
                spectrum = self.volume[x, y, :]

                # Update the spectrum plot
                self.spectrum.set_ydata(spectrum)
                if self.fitted is not None:
                    spectrum_fitted = self.fitted[x, y, :]
                    self.spectrum_fitted.set_ydata(spectrum_fitted)

                self.ax_spectrum.relim()
                self.ax_spectrum.autoscale_view()

                # Set title with coordinates
                self.ax_spectrum.set_title(f'Histogram (x={x}, y={y})')

        elif event.inaxes == self.ax_spectrum:
            if not self.spectrum_real_time_update:
                return
            index = int(event.xdata)

            # Get the image from the volume
            image = self.volume[:, :, index]

            # Update the image display
            self.image.set_data(image.T)
            self.ax_image.relim()
            self.ax_image.autoscale_view()

            # Set title with bin
            self.ax_image.set_title(f'Image (Bin={index})')

            # Set color limits based on the current image
            self.image.set_clim(np.min(image), np.max(image))

        self.fig.canvas.draw()

    def toggle_real_time(self, event):
        """
        Toggle real-time updates for image and spectrum panels on right-click.

        Right-click over the image disables/enables updates from hovering on it.
        Same applies for the spectrum panel.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Mouse click event used to toggle interactivity.
        """
        if event.button == 3:
            if event.inaxes == self.ax_image:
                self.image_real_time_update = not self.image_real_time_update
            elif event.inaxes == self.ax_spectrum:
                self.spectrum_real_time_update = not self.spectrum_real_time_update