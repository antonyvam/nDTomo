# -*- coding: utf-8 -*-

"""
pencilbeam: Simulation of 2D Pencil Beam CT Data Using Various Acquisition Strategies

This module provides a set of functions to simulate 2D pencil beam computed tomography (CT) 
scanning using different acquisition strategies. The scanning methods include stepped, 
continuous, and zigzag/zigzig scanning patterns, with either rotation or translation as 
the fast axis. These simulations are useful for studying scanning efficiencies, 
sinogram generation, and reconstruction approaches in pencil beam CT.

Available Acquisition Strategies:

1. **Stepped Scans:**
   - **zigzig_stepped_translation**: Zigzig scan with translation as the primary movement axis, moving in discrete steps.
   - **zigzig_stepped_rotation**: Zigzig scan with rotation as the primary movement axis, moving in discrete steps.
   - **zigzag_stepped_translation**: Zigzag scan with translation as the primary movement axis, moving in discrete steps.
   - **zigzag_stepped_rotation**: Zigzag scan with rotation as the primary movement axis, moving in discrete steps.

2. **Fast-Axis Scans (Continuous Motion):**
   - **zigzig_fast_translation**: Zigzig scan with translation as the primary movement axis, moving in continuous motion.
   - **zigzig_fast_rotation**: Zigzig scan with rotation as the primary movement axis, moving in continuous motion.
   - **zigzag_fast_translation**: Zigzag scan with translation as the primary movement axis, moving in continuous motion.
   - **zigzag_fast_rotation**: Zigzag scan with rotation as the primary movement axis, moving in continuous motion.

3. **Continuous Scanning:**
   - **continuous_rot_trans**: Simulates a scan where rotation and translation occur 
     simultaneously. The sample rotates continuously over a full 360° while translating.

Each of these functions generates an animated visualization of the scanning process, 
demonstrating the movement of the sample, the sinogram formation, and the beam behavior 
(turning on/off in stepped scans). These simulations can be useful for testing scanning 
strategies, optimizing acquisition times, and developing reconstruction methods for 
pencil beam CT applications.

@author: Dr A. Vamvakeros
"""

#%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import rotate
from scipy.ndimage import shift


def zigzig_stepped_translation(image, angles, interval=500, cmap="jet"):
    """
    Zigzig scan with translation as the primary movement axis, moving in discrete steps.

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
    total_frames = num_angles * npix * 2 * 2  # ✅ Double frames per position due to beam ON/OFF + zigzag return

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
        rotation_index = frame // (npix * 4)  # Which rotation step
        step_within_angle = frame % (npix * 4)  # Zigzag motion
        translation_step = (step_within_angle // 2) % npix  # Every 2 frames = 1 translation step

        beam_on = (step_within_angle % 2 == 1)  # Beam ON every 2nd frame

        rotation_angle = angles[rotation_index]
        rotated_image = rotate(image_original, rotation_angle, resize=False, mode='constant', cval=0, order=1)

        # Ensure rotated image is exactly npix x npix
        rotated_image = rotated_image[:npix, :npix]

        # Determine ZIG or ZAG movement
        if step_within_angle < npix * 2:
            shift_amount = translation_step  # Forward movement (ZIG)
        else:
            shift_amount = npix - translation_step - 1  # Backward movement (ZAG)

        # Translate the rotated image
        translated_image_canvas = np.zeros_like(image_canvas)
        translated_image_canvas[shift_amount:shift_amount + npix, :] = rotated_image[:npix, :]

        # Beam ON → Update sinogram ONLY during forward motion
        if beam_on and step_within_angle < npix * 2:
            sinogram_canvas[int(npix / 2) + shift_amount, rotation_index] = np.sum(rotated_image[npix - shift_amount - 1, :]) / sf
            translated_image_canvas[npix - 1, :] = sf_im * 1.1  # Beam ON indicator

        # Concatenate image and sinogram
        updated_frame = np.concatenate((translated_image_canvas, space, sinogram_canvas), axis=1)

        # Update the displayed image
        im.set_array(updated_frame)
        im.set_clim(updated_frame.min(), updated_frame.max())  # Adjust color scale

        if step_within_angle < npix * 2:  # Only update during the zig (forward) phase
            beam_status = f"Beam {'ON' if beam_on else 'OFF'}"
        else:
            beam_status = "Beam OFF (Return)"

        ax.set_title(f"Rotation: {rotation_angle:.1f}°, Translation: {shift_amount} pixels, {beam_status}")
        return im,

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=interval)

    # Close the static figure to prevent unwanted display
    plt.close(fig)

    return ani, ani.to_jshtml()


def zigzig_stepped_translation_optimised(image, angles, interval=500, cmap="jet"):
    """
    Zigzig scan with translation as the primary movement axis, moving in discrete steps.

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


def zigzig_stepped_rotation(image, angles, interval=50, cmap="jet"):
    """
    Zigzig scan with rotation as the primary movement axis, moving in discrete steps.

    Parameters
    ----------
    image : np.ndarray
        The original 2D image to be rotated and translated.
    angles : np.ndarray
        A list of angles (in degrees) defining the rotation path.
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
    total_frames = num_angles * npix * 4  # 2 frames per step (ON/OFF) * zig (180°) + zag (return 180°)

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

        # Identify translation step, beam ON/OFF state, and rotation direction
        translation_step = frame // (num_angles * 4)  # Translation index
        step_within_angle = frame % (num_angles * 4)  # Tracks each step within one translation step
        beam_on = (step_within_angle % 2 == 1)  # Beam ON every second frame
        zigzag_phase = (step_within_angle // (2 * num_angles))  # 0 for zig, 1 for zag
        rotation_index = (step_within_angle // 2) % num_angles  # Rotation index for zig/zag

        # Zig: Rotate 0° → 180° while translating with stepped scan
        if zigzag_phase == 0:
            rotation_angle = angles[rotation_index]  # Forward rotation
        # Zag: Rotate 180° → 0° while translating with stepped scan
        else:
            rotation_angle = angles[-(rotation_index + 1)]  # Reverse rotation

        rotated_image = rotate(image_original, rotation_angle, resize=False, mode='constant', cval=0, order=1)

        # Translate image
        shift_amount = translation_step
        translated_image_canvas = np.zeros_like(image_canvas)
        translated_image_canvas[shift_amount:npix + shift_amount, :] = rotated_image

        # Beam indicator (ON only during ON phase)
        if beam_on and zigzag_phase == 0:  # Beam ON only during forward rotation
            translated_image_canvas[npix - 1, :] = sf_im * 1.1  # Beam ON marker

        # Update sinogram only during zig phase and beam ON
        if beam_on and zigzag_phase == 0:
            sinogram_canvas[int(npix / 2) + translation_step, rotation_index] = np.sum(rotated_image[npix - shift_amount - 1, :]) / sf

        # Concatenate image and sinogram
        updated_frame = np.concatenate((translated_image_canvas, space, sinogram_canvas), axis=1)

        # Update display
        im.set_array(updated_frame)
        im.set_clim(updated_frame.min(), updated_frame.max())

        # Title: Show beam only during zig phase when ON
        beam_status = "Beam ON" if beam_on and zigzag_phase == 0 else "Beam OFF"
        ax.set_title(f"Rotation: {rotation_angle:.1f}°, Translation: {shift_amount} pixels, {beam_status}")

        return im,

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=interval)

    # Close the static figure to prevent unwanted display
    plt.close(fig)

    return ani, ani.to_jshtml()


def zigzag_stepped_translation(image, angles, interval=500, cmap="jet"):
    """
    Zigzag scan with translation as the primary movement axis, moving in discrete steps.

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



def zigzag_stepped_rotation(image, angles, interval=50, cmap="jet"):
    """
    Zigzag scan with rotation as the primary movement axis, moving in discrete steps.

    Parameters
    ----------
    image : np.ndarray
        The original 2D image to be rotated and translated.
    angles : np.ndarray
        A list of angles (in degrees) defining the rotation path.
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
    total_frames = num_angles * npix * 2  # Stepped scan: 2 frames per translation step (beam OFF/ON)

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

        # Identify translation step, beam ON/OFF state, and rotation direction
        translation_step = frame // (num_angles * 2)  # Which translation step
        step_within_angle = frame % (num_angles * 2)  # Tracks each step within one translation step
        beam_on = step_within_angle % 2 == 1  # Beam ON every second frame
        rotation_index = step_within_angle // 2  # Rotation index in zig/zag

        # Determine rotation direction (zig or zag)
        if translation_step % 2 == 0:
            rotation_angle = angles[rotation_index]  # Normal rotation (0 to max)
        else:
            rotation_angle = angles[-rotation_index - 1]  # Reverse rotation (max to 0)

        # Rotate the image
        rotated_image = rotate(image_original, rotation_angle, resize=False, mode='constant', cval=0, order=1)

        # Translate the rotated image
        shift_amount = translation_step
        translated_image_canvas = np.zeros_like(image_canvas)
        translated_image_canvas[shift_amount:npix + shift_amount, :] = rotated_image

        # Beam indicator (ON only in ON phase)
        if beam_on:
            translated_image_canvas[npix - 1, :] = sf_im * 1.1  # Beam ON marker

        # Update sinogram only when beam is ON
        if beam_on:
            if translation_step % 2 == 0:
                sinogram_canvas[int(npix / 2) + translation_step, rotation_index] = np.sum(rotated_image[npix - shift_amount - 1, :]) / sf
            else:
                sinogram_canvas[int(npix / 2) + translation_step, -rotation_index] = np.sum(rotated_image[npix - shift_amount - 1, :]) / sf

        # Concatenate image and sinogram
        updated_frame = np.concatenate((translated_image_canvas, space, sinogram_canvas), axis=1)

        # Update display
        im.set_array(updated_frame)
        im.set_clim(updated_frame.min(), updated_frame.max())

        ax.set_title(f"Rotation: {rotation_angle:.1f}°, Translation: {shift_amount} pixels, Beam {'ON' if beam_on else 'OFF'}")

        return im,

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=interval)

    # Close the static figure to prevent unwanted display
    plt.close(fig)

    return ani, ani.to_jshtml()


def zigzig_fast_translation(image, angles, interval=500, cmap="jet"):
    """
    Zigzig scan with translation as the primary movement axis, moving in continuous motion.

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


def zigzig_fast_rotation(image, angles, interval=500, cmap="jet"):
    """
    Zigzig scan with rotation as the primary movement axis, moving in continuous motion.

    Parameters
    ----------
    image : np.ndarray
        The original 2D image to be rotated and translated.
    angles : np.ndarray
        A list of angles (in degrees) defining the rotation path.
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
    total_frames = num_angles * npix * 2  # Each translation step has a zig (0-180°) and zag (180-0°)

    print(f"Total frames: {total_frames}")

    image_original = image.copy()

    # Create canvases
    sinogram_canvas = np.zeros((2 * npix, num_angles), dtype='float32')
    image_canvas = np.zeros((2 * npix, npix), dtype='float32')
    image_canvas[0:npix, :] = image_original

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

        # Determine translation step and zig/zag phase
        translation_step = frame // (num_angles * 2)  # Translation index
        rotation_index = (frame % (num_angles * 2))  # Within each translation step, rotation zig/zag
        zigzag_phase = rotation_index // num_angles  # 0 for zig, 1 for zag
        angle_index = rotation_index % num_angles  # Angle index within zig or zag

        # Zig: Rotate 0° → 180° while translating down with beam on
        if zigzag_phase == 0:
            rotation_angle = angles[angle_index]  # Forward rotation
            beam_on = True
        # Zag: Rotate 180° → 0° while translating down with beam off
        else:
            rotation_angle = angles[-(angle_index + 1)]  # Reverse rotation
            beam_on = False

        rotated_image = rotate(image_original, rotation_angle, resize=False, mode='constant', cval=0, order=1)

        # Translate image
        shift_amount = translation_step
        translated_image_canvas = np.zeros_like(image_canvas)
        translated_image_canvas[shift_amount:npix + shift_amount, :] = rotated_image

        # Beam indicator (ON only during zig)
        if beam_on:
            translated_image_canvas[npix - 1, :] = sf_im * 1.1  # Beam ON marker

        # Update sinogram only during zig phase
        if beam_on:
            sinogram_canvas[int(npix / 2) + shift_amount, angle_index] = np.sum(rotated_image[npix - shift_amount - 1, :]) / sf

        # Concatenate image and sinogram
        updated_frame = np.concatenate((translated_image_canvas, space, sinogram_canvas), axis=1)

        # Update display
        im.set_array(updated_frame)
        im.set_clim(updated_frame.min(), updated_frame.max())

        ax.set_title(f"Rotation: {rotation_angle:.1f}°, Translation: {shift_amount} pixels, Beam {'ON' if beam_on else 'OFF'}")

        return im,

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=interval)

    # Close the static figure to prevent unwanted display
    plt.close(fig)

    return ani, ani.to_jshtml()


def zigzag_fast_translation(image, angles, interval=500, cmap="jet"):
    """
    Zigzag scan with translation as the primary movement axis, moving in continuous motion.

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


def zigzag_fast_rotation(image, angles, interval=50, cmap="jet"):
    """
    Zigzag scan with rotation as the primary movement axis, moving in continuous motion.

    Parameters
    ----------
    image : np.ndarray
        The original 2D image to be rotated and translated.
    angles : np.ndarray
        A list of angles (in degrees) defining the rotation path.
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
    total_frames = num_angles * npix  # Each translation step scans across all angles

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

        # Identify translation step and rotation direction
        translation_step = frame // num_angles  # Which translation step
        rotation_index = frame % num_angles  # Current rotation index

        # Determine rotation direction (zig or zag)
        if translation_step % 2 == 0:
            rotation_angle = angles[rotation_index]  # Normal rotation (0 to max)
        else:
            rotation_angle = angles[-rotation_index - 1]  # Reverse rotation (max to 0)

        # Rotate the image
        rotated_image = rotate(image_original, rotation_angle, resize=False, mode='constant', cval=0, order=1)

        # Translate the rotated image
        shift_amount = translation_step
        translated_image_canvas = np.zeros_like(image_canvas)
        translated_image_canvas[shift_amount:npix + shift_amount, :] = rotated_image

        translated_image_canvas[npix - 1, :] = sf_im * 1.1  # Beam ON indicator

        # Sinogram updates continuously
        if translation_step % 2 == 0:
            sinogram_canvas[int(npix / 2) + translation_step, rotation_index] = np.sum(rotated_image[npix - shift_amount - 1, :]) / sf
        else:
            sinogram_canvas[int(npix / 2) + translation_step, -rotation_index] = np.sum(rotated_image[npix - shift_amount - 1, :]) / sf

        # Concatenate image and sinogram
        updated_frame = np.concatenate((translated_image_canvas, space, sinogram_canvas), axis=1)

        # Update display
        im.set_array(updated_frame)
        im.set_clim(updated_frame.min(), updated_frame.max())

        ax.set_title(f"Rotation: {rotation_angle:.1f}°, Translation: {shift_amount} pixels")

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
    Simulate a simultaneous continuous rotation and translation scan with the beam always ON.

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
    image_canvas[0:npix, :] = image_original

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

        # Rotate the image
        rotated_image = rotate(image_original, rotation_angle, resize=False, mode='constant', cval=0, order=1)

        # Translate the rotated image
        shift_amount = translation_step
        translated_image_canvas = np.zeros_like(image_canvas)
        translated_image_canvas[shift_amount:npix + shift_amount, :] = rotated_image

        translated_image_canvas[npix - 1, :] = sf_im * 1.1  # Beam ON indicator

        # Sinogram updates continuously
        sinogram_canvas[int(npix / 2) + translation_step, rotation_index] = np.sum(rotated_image[npix - shift_amount - 1, :]) / sf

        # Concatenate image and sinogram
        updated_frame = np.concatenate((translated_image_canvas, space, sinogram_canvas), axis=1)

        # Update display
        im.set_array(updated_frame)
        im.set_clim(updated_frame.min(), updated_frame.max())

        ax.set_title(f"Rotation: {rotation_angle:.1f}°, Translation: {shift_amount} pixels")

        return im,

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=interval)

    # Close the static figure to prevent unwanted display
    plt.close(fig)

    return ani, ani.to_jshtml()


def continuous_rot_trans_optimised(image, angles, num_trans_steps, interval=50, cmap="jet"):
    """
    Simulate a simultaneous continuous rotation and translation scan with cubic interpolation.

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