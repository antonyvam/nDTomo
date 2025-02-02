# -*- coding: utf-8 -*-

"""
pencilbeam: Simulation of 2D pencil beam CT data using various acquisition strategies

@author: Dr A. Vamvakeros
"""

#%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import rotate
from skimage.transform import rotate
from scipy.ndimage import shift

def zigzag(image, angles, interval=500, cmap="jet"):
    """
    Simulate a zigzag scan with rotation and translation while dynamically updating the sinogram.

    Parameters
    ----------
    image : np.ndarray
        The original 2D image to be rotated.
    angles : list or np.ndarray
        1D array of angles (in degrees) for rotation at each frame.
    interval : int, optional
        Time between frames in milliseconds (default is 500ms).
    cmap : str, optional
        Colormap for displaying images (default is 'jet').

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
        The created animation.
    ani_html : str
        HTML representation of the animation for display in Jupyter.
    """

    npix = image.shape[0]
    num_angles = len(angles)
    total_frames = num_angles * npix  # Each angle goes through npix translations

    print(f"Total frames: {total_frames}")

    image_original = image.copy()
    
    # Create canvases
    sinogram = np.zeros((npix, num_angles), dtype='float32')  # Sinogram accumulates data
    sinogram_canvas = np.zeros((2 * npix, num_angles), dtype='float32')
    image_canvas = np.zeros((2 * npix, npix), dtype='float32')
    image_canvas[0:npix, :] = image_original

    space = np.zeros((2 * npix, 10), dtype='float32')
    new_image = np.concatenate((image_canvas, space, sinogram_canvas), axis=1)

    tmp = np.sum(image, axis=1)
    sf = np.max(tmp)
    sf_im = np.max(image)

    new_image[npix,:] = sf_im*1.1

    # Set up figure
    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.imshow(new_image, cmap=cmap, animated=True)
    

    def update(frame):
        """
        Update function for the animation.
        """
        rotation_index = frame // npix  # Which rotation step
        translation_step = frame % npix  # Translation step within that rotation

        rotation_angle = angles[rotation_index]
        rotated_image = rotate(image_original, rotation_angle, resize=False, mode='constant', cval=0, order=1)

        # Zigzag movement logic
        if rotation_index % 2 == 0:
            shift_amount = translation_step  # Move downward
        else:
            shift_amount = npix - translation_step - 1  # Move upward

        # Translate the rotated image
        translated_image_canvas = np.zeros_like(image_canvas)
        translated_image_canvas[shift_amount:npix + shift_amount, :] = rotated_image
        translated_image_canvas[npix-1,:] = sf_im*1.1

        # Update sinogram
        if rotation_index % 2 == 0:
            sinogram_canvas[int(npix/2) + translation_step, rotation_index] = np.sum(rotated_image[npix - translation_step - 1,:]) / sf
        else:
            sinogram_canvas[3*int(npix/2) - translation_step, rotation_index] = np.sum(rotated_image[translation_step,:]) / sf


        # Concatenate image and sinogram
        updated_frame = np.concatenate((translated_image_canvas, space, sinogram_canvas), axis=1)

        # Update the displayed image
        im.set_array(updated_frame)
        # im.set_clim(updated_frame.min(), updated_frame.max())  # Adjust color scale
        ax.set_title(f"Rotation: {rotation_angle:.1f}°, Translation: {shift_amount} pixels")

        return im,

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=interval)

    # Close the static figure to prevent unwanted display
    plt.close(fig)

    return ani, ani.to_jshtml()

def zigzig(image, angles, interval=500, cmap="jet"):
    """
    Simulate a zigzig scan with rotation and translation while dynamically updating the sinogram.

    Parameters
    ----------
    image : np.ndarray
        The original 2D image to be rotated.
    angles : list or np.ndarray
        1D array of angles (in degrees) for rotation at each frame.
    interval : int, optional
        Time between frames in milliseconds (default is 500ms).
    cmap : str, optional
        Colormap for displaying images (default is 'jet').

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
        The created animation.
    ani_html : str
        HTML representation of the animation for display in Jupyter.
    """

    npix = image.shape[0]
    num_angles = len(angles)
    total_frames = num_angles * npix * 2  # Each angle goes through npix translations + return

    print(f"Total frames: {total_frames}")

    image_original = image.copy()
    
    # Create canvases
    sinogram = np.zeros((npix, num_angles), dtype='float32')  # Sinogram accumulates data
    sinogram_canvas = np.zeros((2 * npix, num_angles), dtype='float32')
    image_canvas = np.zeros((2 * npix, npix), dtype='float32')
    image_canvas[0:npix, :] = image_original

    space = np.zeros((2 * npix, 10), dtype='float32')
    new_image = np.concatenate((image_canvas, space, sinogram_canvas), axis=1)

    tmp = np.sum(image, axis=1)
    sf = np.max(tmp)
    sf_im = np.max(image)

    new_image[npix, :] = sf_im * 1.1  # Beam line indicator (only for forward pass)

    # Set up figure
    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.imshow(new_image, cmap=cmap, animated=True)

    def update(frame):
        """
        Update function for the animation.
        """
        rotation_index = frame // (npix * 2)  # Which rotation step
        translation_step = (frame % (npix * 2))  # Forward & Return Phase

        rotation_angle = angles[rotation_index]
        rotated_image = rotate(image_original, rotation_angle, resize=False, mode='constant', cval=0, order=1)

        # Forward scan with beam ON (sinogram updates)
        if translation_step < npix:
            shift_amount = translation_step
            beam_on = True
        # Return scan with beam OFF (no sinogram update)
        else:
            shift_amount = (2 * npix - translation_step - 1)
            beam_on = False

        # Translate the rotated image
        translated_image_canvas = np.zeros_like(image_canvas)
        translated_image_canvas[shift_amount:npix + shift_amount, :] = rotated_image
        
        # Beam indicator (ON only in forward phase)
        if beam_on:
            translated_image_canvas[npix-1, :] = sf_im * 1.1  # Mid row shows beam

        # Update sinogram only in forward pass
        if beam_on:
            sinogram_canvas[int(npix/2) + shift_amount, rotation_index] = np.sum(rotated_image[npix - shift_amount - 1,:]) / sf

        # Concatenate image and sinogram
        updated_frame = np.concatenate((translated_image_canvas, space, sinogram_canvas), axis=1)

        # Update the displayed image
        im.set_array(updated_frame)
        im.set_clim(updated_frame.min(), updated_frame.max())  # Adjust color scale
        ax.set_title(f"Rotation: {rotation_angle:.1f}°, Translation: {shift_amount} pixels, Beam {'ON' if beam_on else 'OFF'}")

        return im,

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=interval)

    # Close the static figure to prevent unwanted display
    plt.close(fig)

    return ani, ani.to_jshtml()

def stepped_zigzig(image, angles, interval=500, cmap="jet"):
    """
    Simulate a stepped zigzig scan with rotation and translation where each translation step has two frames 
    (one without beam, one with beam) while dynamically updating the sinogram.

    Parameters
    ----------
    image : np.ndarray
        The original 2D image to be rotated.
    angles : list or np.ndarray
        1D array of angles (in degrees) for rotation at each frame.
    interval : int, optional
        Time between frames in milliseconds (default is 500ms).
    cmap : str, optional
        Colormap for displaying images (default is 'jet').

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
        The created animation.
    ani_html : str
        HTML representation of the animation for display in Jupyter.
    """


    npix = image.shape[0]
    num_angles = len(angles)
    total_frames = num_angles * npix * 3 # Zig (2*npix frames) + Zag (npix frames)

    image_original = image.copy()

    # Create canvases
    sinogram = np.zeros((npix, num_angles), dtype='float32')  # Sinogram accumulates data
    sinogram_canvas = np.zeros((2 * npix, num_angles), dtype='float32')
    image_canvas = np.zeros((2 * npix, npix), dtype='float32')
    image_canvas[:npix, :] = image_original

    space = np.zeros((2 * npix, 10), dtype='float32')
    new_image = np.concatenate((image_canvas, space, sinogram_canvas), axis=1)

    tmp = np.sum(image, axis=1)
    sf = np.max(tmp)
    sf_im = np.max(image)

    # Set up figure
    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.imshow(new_image, cmap=cmap, animated=True)


    def update(frame):
        """
        Update function for the animation.
        """

        # Identify which phase (zig/zag) we're in
        step_within_angle = frame % (npix * 3)  # 2*npix for zig + npix for continuous zag
        rotation_index = frame // (npix * 3)  # Current rotation step
        rotation_angle = angles[rotation_index]

        rotated_image = rotate(image_original, rotation_angle, resize=False, mode='constant', cval=0, order=1)
        # Ensure rotated image is exactly npix x npix
        rotated_image = rotated_image[:npix, :npix]

        # Zig: Step-wise movement (beam ON/OFF per position)
        if step_within_angle < npix * 2:
            translation_step = step_within_angle // 2
            beam_on = step_within_angle % 2 == 1  # Beam ON every 2nd frame

            # Translate image
            shift_amount = translation_step
            translated_image_canvas = np.zeros_like(image_canvas)
            translated_image_canvas[shift_amount:npix + shift_amount, :] = rotated_image

            # Sinogram updates **only when beam is ON**
            if beam_on:
                sinogram_canvas[int(npix / 2) + translation_step, rotation_index] = np.sum(rotated_image[npix - shift_amount - 1, :]) / sf
                translated_image_canvas[npix - 1, :] = sf_im * 1.1  # Beam ON indicator

        # Zag: **Continuous return (No sinogram updates)**
        else:
            return_step = step_within_angle - (npix * 2)  # Range: [0, npix-1]
            shift_amount = npix - return_step - 1  # Smooth return

            # Translate image continuously back
            translated_image_canvas = np.zeros_like(image_canvas)
            translated_image_canvas[shift_amount:npix + shift_amount, :] = rotated_image

            # **Beam always OFF, No sinogram updates**
            beam_on = False

        # Concatenate image and sinogram
        updated_frame = np.concatenate((translated_image_canvas, space, sinogram_canvas), axis=1)

        # Update display
        im.set_array(updated_frame)
        im.set_clim(updated_frame.min(), updated_frame.max())

        # Title: Show beam only during zig phase
        if step_within_angle < npix * 2:
            beam_status = f"Beam {'ON' if beam_on else 'OFF'}"
        else:
            beam_status = "Beam OFF (Return - Continuous)"

        ax.set_title(f"Rotation: {rotation_angle:.1f}°, Translation: {shift_amount} pixels, {beam_status}")

        return im,

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=interval)

    # Close the static figure to prevent unwanted display
    plt.close(fig)

    return ani, ani.to_jshtml()

def stepped_zigzag(image, angles, interval=500, cmap="jet"):
    """
    Simulate a stepped zigzag scan with rotation and translation while dynamically updating the sinogram.

    Parameters
    ----------
    image : np.ndarray
        The original 2D image to be rotated.
    angles : list or np.ndarray
        1D array of angles (in degrees) for rotation at each frame.
    interval : int, optional
        Time between frames in milliseconds (default is 500ms).
    cmap : str, optional
        Colormap for displaying images (default is 'jet').

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
        The created animation.
    ani_html : str
        HTML representation of the animation for display in Jupyter.
    """

    npix = image.shape[0]
    num_angles = len(angles)
    total_frames = num_angles * npix * 2  # Stepped scan: 2 frames per translation step (beam OFF/ON)

    print(f"Total frames: {total_frames}")

    image_original = image.copy()

    # Create canvases
    sinogram = np.zeros((npix, num_angles), dtype='float32')  # Sinogram accumulates data
    sinogram_canvas = np.zeros((2 * npix, num_angles), dtype='float32')
    image_canvas = np.zeros((2 * npix, npix), dtype='float32')
    image_canvas[0:npix, :] = image_original

    space = np.zeros((2 * npix, 10), dtype='float32')
    new_image = np.concatenate((image_canvas, space, sinogram_canvas), axis=1)

    tmp = np.sum(image, axis=1)
    sf = np.max(tmp)
    sf_im = np.max(image)

    new_image[npix, :] = sf_im * 1.1  # Beam indicator

    # Set up figure
    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.imshow(new_image, cmap=cmap, animated=True)

    def update(frame):
        """
        Update function for the animation.
        """
        rotation_index = frame // (npix * 2)  # Which rotation step
        step_within_angle = frame % (npix * 2)  # Tracks each step within one angle

        translation_step = step_within_angle // 2  # Two frames per translation step
        beam_on = step_within_angle % 2 == 1  # Beam ON every second frame

        rotation_angle = angles[rotation_index]
        rotated_image = rotate(image_original, rotation_angle, resize=False, mode='constant', cval=0, order=1)

        # Zigzag movement logic
        if rotation_index % 2 == 0:
            shift_amount = translation_step  # Move downward
        else:
            shift_amount = npix - translation_step - 1  # Move upward

        # Translate the rotated image
        translated_image_canvas = np.zeros_like(image_canvas)
        translated_image_canvas[shift_amount:npix + shift_amount, :] = rotated_image

        # Sinogram updates **only when beam is ON**
        if beam_on:
            if rotation_index % 2 == 0:
                sinogram_canvas[int(npix / 2) + translation_step, rotation_index] = np.sum(rotated_image[npix - shift_amount - 1, :]) / sf
            else:
                sinogram_canvas[3 * int(npix / 2) - translation_step, rotation_index] = np.sum(rotated_image[translation_step, :]) / sf
            translated_image_canvas[npix - 1, :] = sf_im * 1.1  # Beam ON indicator

        # Concatenate image and sinogram
        updated_frame = np.concatenate((translated_image_canvas, space, sinogram_canvas), axis=1)

        # Update the displayed image
        im.set_array(updated_frame)
        im.set_clim(updated_frame.min(), updated_frame.max())  # Adjust color scale
        ax.set_title(f"Rotation: {rotation_angle:.1f}°, Translation: {shift_amount} pixels, Beam {'ON' if beam_on else 'OFF'}")

        return im,

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=interval)

    # Close the static figure to prevent unwanted display
    plt.close(fig)

    return ani, ani.to_jshtml()


def translate_image(image, shift_x, shift_y, subpixel=True, method="interp"):
    """
    Translate a 2D image by (shift_x, shift_y) pixels using interpolation.

    Parameters
    ----------
    image : np.ndarray
        The input 2D image.
    shift_x : float
        Shift in the x-direction.
    shift_y : float
        Shift in the y-direction.
    subpixel : bool, optional
        If True, allows subpixel translations using interpolation.
    method : str, optional
        - "interp": Uses `scipy.ndimage.shift` (cubic interpolation).

    Returns
    -------
    translated_image : np.ndarray
        The translated image.
    """
    if subpixel:
        return shift(image, shift=(shift_y, shift_x), mode='constant', cval=0, order=3)  # Cubic interpolation
    else:
        shift_x = int(round(shift_x))
        shift_y = int(round(shift_y))
        translated_image = np.zeros_like(image)
        
        h, w = image.shape
        x_start, x_end = max(0, shift_x), min(w, w + shift_x)
        y_start, y_end = max(0, shift_y), min(h, h + shift_y)

        translated_image[y_start:y_end, x_start:x_end] = image[
            y_start - shift_y : y_end - shift_y, x_start - shift_x : x_end - shift_x
        ]
        
        return translated_image

def continuous_rot_trans(image, angles, num_trans_steps, interval=50, cmap="jet"):
    """
    Simulate a simultaneous continuous rotation and vertical translation scan with cubic interpolation.

    Parameters
    ----------
    image : np.ndarray
        The original 2D image to be rotated and translated.
    angles : np.ndarray
        A list of angles (in degrees) defining the rotation path.
    num_trans_steps : int
        Number of translation steps for one full rotation cycle.
    interval : int, optional
        Time between frames in milliseconds (default is 50ms).
    cmap : str, optional
        Colormap for displaying images (default is 'jet').

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
        The created animation.
    ani_html : str
        HTML representation of the animation for display in Jupyter.
    """

    npix = image.shape[0]
    num_angles = len(angles)
    total_frames = num_trans_steps * num_angles  # Each translation step spans all angles

    print(f"Total frames: {total_frames}")

    image_original = image.copy()

    # Create canvases
    sinogram_canvas = np.zeros((2 * npix, num_angles), dtype='float32')
    image_canvas = np.zeros((2 * npix, npix), dtype='float32')
    image_canvas[:npix, :] = image_original

    space = np.zeros((2 * npix, 10), dtype='float32')
    new_image = np.concatenate((image_canvas, space, sinogram_canvas), axis=1)

    tmp = np.sum(image, axis=1)
    sf = np.max(tmp)
    sf_im = np.max(image)

    # Set up figure
    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.imshow(new_image, cmap=cmap, animated=True)

    def update(frame):
        """
        Update function for the animation.
        """

        # Determine translation step and rotation angle
        translation_step = frame // num_angles  # Translation index
        rotation_index = frame % num_angles  # Rotation index within the angles list
        rotation_angle = angles[rotation_index]  # Current rotation angle

        # Compute **subpixel translation** (ensuring smooth movement)
        subpixel_shift = (translation_step + (rotation_index / num_angles))  

        # Rotate the image
        rotated_image = rotate(image_original, rotation_angle, resize=False, mode='constant', cval=0, order=1)

        # Translate the rotated image **correctly using cubic interpolation**

        translated_image_canvas = np.zeros_like(image_canvas)
        translated_image_canvas[:npix, :] = rotated_image  # Insert translated image
        translated_image_canvas = translate_image(translated_image_canvas, shift_x=0, shift_y=subpixel_shift, subpixel=True, method="interp")
        translated_image_canvas[npix - 1, :] = sf_im * 1.1  # Beam ON indicator

        # Sinogram updates continuously
        sinogram_canvas[int(npix / 2) + translation_step, rotation_index] = np.sum(rotated_image[npix - translation_step - 1, :]) / sf

        # Concatenate image and sinogram
        updated_frame = np.concatenate((translated_image_canvas, space, sinogram_canvas), axis=1)

        # Update display
        im.set_array(updated_frame)
        # im.set_clim(updated_frame.min(), updated_frame.max())
        im.set_clim(0, updated_frame.max())

        ax.set_title(f"Rotation: {rotation_angle:.1f}°, Translation: {subpixel_shift:.2f} pixels")

        return im,

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=interval)

    # Close the static figure to prevent unwanted display
    plt.close(fig)

    return ani, ani.to_jshtml()