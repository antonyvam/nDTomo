# -*- coding: utf-8 -*-
"""
nDTomo Segmentation
====================

2D multi-scale featurisation of a single channel image. Taken from 'interactive-seg-backend',
url: https://github.com/tldr-group/interactive-seg-backend

Approach inspired by (1)
https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.multiscale_basic_features
Designed to be a Python equivalent of (most) of the features present at (2) https://imagej.net/plugins/tws/
Heavy use of skimage filters, filters.rank and feature.
General approach is:
• for each $sigma (a scale over which to compute a feature for a pixel):
    • compute each singlescale singlechannel feature
• compute scale free features (difference of Gaussians, Membrane Projections, Bilateral)
• combine, stack as np array in form (HxWxN_features)

Singlescale feature computation is mapped over multiple threads as in (1).
Every feature computes a value for *every pixel* in the image.

@author: Ronan Docherty
"""

import numpy as np
from numpy import log2, logspace
import numpy.typing as npt

from skimage import filters
from skimage.draw import disk  # type: ignore[import]
from scipy.ndimage import convolve  # type: ignore[import]
from skimage.util.dtype import img_as_float32  # type: ignore[import]
from scipy.ndimage import rotate  # type: ignore[import]
from skimage.filters import gaussian  # type: ignore[import]

from itertools import combinations_with_replacement
from itertools import chain

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

from xgboost import XGBClassifier

from tifffile import imread
from PIL import Image
from os.path import exists

from dataclasses import dataclass, asdict
from json import dumps
from typing import Literal, cast, TypeAlias


# %% ===================================TYPES===================================

FloatArr: TypeAlias = npt.NDArray[np.float16] | npt.NDArray[np.float32] | npt.NDArray[np.float64]
UInt8Arr: TypeAlias = npt.NDArray[np.uint8]
IntArr: TypeAlias = npt.NDArray[np.uint8] | npt.NDArray[np.uint16] | npt.NDArray[np.int32] | npt.NDArray[np.int64]
Arr: TypeAlias = FloatArr | IntArr


@dataclass
class FeatureConfig:
    """Set of (classical) features for the image, & the length scales to apply them over."""

    name: str = "default"
    desc: str = "weka-style features"

    # weka has a 0.4 * multiplier before its $sigma for the gaussian blurs
    add_weka_sigma_multiplier: bool = True
    # gaussian blur with std=$sigma
    gaussian_blur: bool = True
    # gradient magnitude (of gaussian blur $sigma)
    sobel_filter: bool = True
    # hessian (of gaussian blur $sigma) - either eigs OR eigs + mod, trace, det
    hessian_filter: bool = True
    # difference of all gaussians based on set $sigmas
    difference_of_gaussians: bool = True
    # membrane = convolve img with N pixel line kernel rotated R times
    membrane_projections: bool = True
    # mean of neighbours in $sigma radius
    mean: bool = False
    # minimum of neighbours in $sigma radius
    minimum: bool = False
    # maximum of neighbours in $sigma radius
    maximum: bool = False
    # median of neighbours in $sigma radius
    median: bool = False
    # bilateral: mean of pixels with certain greyscale value within certain radius
    bilateral: bool = False

    add_mod_trace_det_hessian: bool = True

    membrane_thickness: int = 1
    membrane_patch_size: int = 15

    # apply features to unblurred img
    add_zero_scale_features: bool = True

    min_sigma: float = 1.0
    max_sigma: float = 16.0
    sigmas: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)

    def __post_init__(self) -> None:
        assert self.min_sigma >= 0, "min_sigma must be greater than (or equal to) 0"
        assert self.max_sigma <= 64, "max_sigma must be less than (or equal to) 64"

        if self.min_sigma != 1.0 or self.max_sigma != 16.0:
            # update sigmas based on min/max if changed
            # NB: to set sigmas explicitly, set it in init and don't adjust min/max
            log_min: float = log2(self.min_sigma)
            log_max: float = log2(self.max_sigma)
            num_sigma = int(log_max - log_min + 1)
            sigmas: tuple[float, ...] = tuple(
                logspace(
                    log_min,
                    log_max,
                    num=num_sigma,
                    base=2,
                    endpoint=True,
                )
            )
            self.sigmas = sigmas

        assert self.membrane_thickness >= 1, "membrane_thickness must be greater than (or equal to) 0"
        assert self.membrane_patch_size >= 3, "membrane_patch_size must be greater than (or equal to) 3"

    def __repr__(self) -> str:
        to_stringify = asdict(self)
        out_str = f"FEATURE CONFIG: \n`{self.name}`: {self.desc}\n" + dumps(to_stringify, ensure_ascii=True, indent=2)
        to_stringify.pop("name")
        to_stringify.pop("desc")
        return out_str


# - 2 to allow for main & gui threads
BACKEND: Literal["loky", "threading"] = "threading"
N_ALLOWED_CPUS = cpu_count() - 2


# %% ===================================HELPER FUNCTIONS===================================
def _make_footprint(sigma: int) -> npt.NDArray[np.uint8]:
    """Return array of zeros with centreed circle of radius sigma set to 1.

    :param sigma: radius of footprint
    :type sigma: int
    :return: array with disk radius sigma set to 1
    :rtype: np.ndarray
    """
    circle_footprint = np.zeros((2 * sigma + 1, 2 * sigma + 1), dtype=np.uint8)
    centre = (sigma, sigma)
    rr: npt.NDArray[np.int64]
    cc: npt.NDArray[np.int64]
    rr, cc = disk(centre, sigma)
    circle_footprint[rr, cc] = 1
    return circle_footprint


def _rotate_ts(
    input: FloatArr,
    angle: float,
    axes: tuple[int, ...] = (1, 0),
    reshape: bool = True,
    order: int = 3,
    mode: str = "constant",
    cval: float = 0,
    prefilter: bool = True,
) -> FloatArr:
    return rotate(input, angle, axes, reshape, None, order, mode, cval, prefilter)  # type: ignore


def _gaussian_ts(
    image: Arr,
    sigma: float = 1,
    mode: str = "nearest",
    cval: int = 0,
    preserve_range: bool = False,
    truncate: float = 4,
    channel_axis: int | None = None,
) -> FloatArr:
    return gaussian(
        image,
        sigma,
        mode=mode,
        cval=cval,
        preserve_range=preserve_range,
        truncate=truncate,
        channel_axis=channel_axis,
    )  # type: ignore


# %% ===================================SINGLESCALE FEATURES===================================


def _singlescale_gaussian(img: FloatArr, sigma: int, mult: float = 1.0) -> FloatArr:
    """Gaussian blur of each pixel in $img of scale/radius $sigma.

    :param img: img arr
    :type img: np.ndarray
    :param sigma: radius for footprint
    :type sigma: int
    :return: filtered array
    :rtype: np.ndarray
    """
    adj_sigma = mult * sigma
    k = 4 * adj_sigma + 1
    trunc = (k + 1) / (2 * adj_sigma)
    out = _gaussian_ts(img, int(adj_sigma), preserve_range=True, truncate=trunc)
    return out


def _singlescale_edges(
    gaussian_filtered: FloatArr,
) -> FloatArr:
    """Sobel filter applied to gaussian filtered arr of scale sigma to detect edges.

    :param gaussian_filtered: img array (that has optionally been gaussian blurred)
    :type gaussian_filtered: np.ndarray
    :return: sobel filtered (edge-detecting) array
    :rtype: np.ndarray
    """
    g_x = np.gradient(gaussian_filtered, axis=-1)
    g_y = np.gradient(gaussian_filtered, axis=-2)
    return np.sqrt(g_x**2 + g_y**2)


def _singlescale_hessian(gaussian_filtered: FloatArr, return_full: bool = True) -> tuple[FloatArr, ...]:
    """Compute mod, trace, det and eigenvalues of Hessian matrix of $gaussian_filtered image (i.e for every pixel).

    :param gaussian_filtered: img array (that has optionally been gaussian blurred)
    :type gaussian_filtered: np.ndarray
    :return: 5 arrays the same shape as input that are the module, trace, determinant and first 2 eigenvalues
        of the hessian at that pixel
    :rtype: Tuple[np.ndarray, ...]
    """
    H_elems = [
        np.gradient(np.gradient(gaussian_filtered)[ax0], axis=ax1)
        for ax0, ax1 in combinations_with_replacement(range(gaussian_filtered.ndim), 2)
    ]
    a, b, d = H_elems
    mod = np.sqrt(a**2 + b**2 + d**2)
    trace = a + d
    det = a * d - b**2
    # orientation_2 = orientation_1 + np.pi / 2
    # eigvals = feature.hessian_matrix_eigvals(H_elems)
    eig1 = trace + np.sqrt((4 * b**2 + (a - d) ** 2))
    eig2 = trace - np.sqrt((4 * b**2 + (a - d) ** 2))

    if return_full:
        return (eig1 / 2.0, eig2 / 2.0, mod, trace, det)
    else:
        return (eig1 / 2.0, eig2 / 2.0)


def _singlescale_mean(
    byte_img: npt.NDArray[np.uint8], sigma_rad_footprint: npt.NDArray[np.uint8]
) -> npt.NDArray[np.uint8]:
    """Mean pixel intensity over footprint $sigma_rad_footprint. Needs img in np.uint8 format.

    :param byte_img: img arr in uint8 format
    :type byte_img: np.ndarray
    :param sigma_rad_footprint: radius of footprint
    :type sigma_rad_footprint: np.ndarray
    :return: mean filtered img
    :rtype: np.ndarray
    """
    out: npt.NDArray[np.uint8] = filters.rank.mean(byte_img, sigma_rad_footprint)  # type: ignore
    return out


def _singlescale_median(
    byte_img: npt.NDArray[np.uint8], sigma_rad_footprint: npt.NDArray[np.uint8]
) -> npt.NDArray[np.uint8]:
    """Median pixel intensity over footprint $sigma_rad_footprint. Needs img in np.uint8 format.

    :param byte_img: img arr in uint8 format
    :type byte_img: np.ndarray
    :param sigma_rad_footprint: radius of footprint
    :type sigma_rad_footprint: np.ndarray
    :return: mean filtered img
    :rtype: np.ndarray
    """
    return filters.rank.median(byte_img, sigma_rad_footprint)  # type: ignore


def _singlescale_maximum(
    byte_img: npt.NDArray[np.uint8], sigma_rad_footprint: npt.NDArray[np.uint8]
) -> npt.NDArray[np.uint8]:
    """maximum pixel intensity over footprint $sigma_rad_footprint. Needs img in np.uint8 format.

    :param byte_img: img arr in uint8 format
    :type byte_img: np.ndarray
    :param sigma_rad_footprint: radius of footprint
    :type sigma_rad_footprint: np.ndarray
    :return: maximum filtered img
    :rtype: np.ndarray
    """
    out: npt.NDArray[np.uint8] = filters.rank.maximum(byte_img, sigma_rad_footprint)  # type: ignore
    return out


def _singlescale_minimum(
    byte_img: npt.NDArray[np.uint8], sigma_rad_footprint: npt.NDArray[np.uint8]
) -> npt.NDArray[np.uint8]:
    """maximum pixel intensity over footprint $sigma_rad_footprint. Needs img in np.uint8 format.

    :param byte_img: img arr in uint8 format
    :type byte_img: np.ndarray
    :param sigma_rad_footprint: radius of footprint
    :type sigma_rad_footprint: np.ndarray
    :return: minimum filtered img
    :rtype: np.ndarray
    """
    out: npt.NDArray[np.uint8] = filters.rank.minimum(byte_img, sigma_rad_footprint)  # type: ignore
    return out


# # %% ===================================SCALE-FREE FEATURES===================================
def _bilateral(byte_img: npt.NDArray[np.uint8]) -> list[npt.NDArray[np.uint8]]:
    """For $sigma in [5, 10], for $value_range in [50, 100],
        compute mean of pixels in $sigma radius inside $value_range window for each pixel.

    :param img: img arr
    :type img: np.ndarray
    :return: bilateral filtered arrs stacked in a single np array
    :rtype: np.ndarray
    """
    bilaterals: list[npt.NDArray[np.uint8]] = []
    for spatial_radius in (5, 10):
        footprint = _make_footprint(spatial_radius)
        for value_range in (50, 100):  # check your pixels are [0, 255]
            filtered: npt.NDArray[np.uint8] = filters.rank.mean_bilateral(  # type: ignore
                byte_img, footprint, s0=value_range, s1=value_range
            )
            bilaterals.append(filtered)
    return bilaterals


def _difference_of_gaussians(
    gaussian_blurs: list[FloatArr],
) -> list[FloatArr]:
    """Compute their difference of each arr in $gaussian_blurs (representing different $sigma scales) with smaller arrs.

    :param gaussian_blurs: list of arrs of img filtered with gaussian blur at different length scales
    :type gaussian_blurs: List[np.ndarray]
    :return: list of differences of each blurred img with smaller length scales.
    :rtype: List[np.ndarray]
    """
    # weka computes dog for  each filter of a *lower* sigma
    dogs: list[FloatArr] = []
    for i in range(len(gaussian_blurs)):
        sigma_1 = gaussian_blurs[i]
        for j in range(i):
            sigma_2 = gaussian_blurs[j]
            dogs.append(sigma_2 - sigma_1)
    return dogs


def _membrane_projections(
    img: FloatArr,
    membrane_patch_size: int = 19,
    membrane_thickness: int = 1,
    num_workers: int | None = N_ALLOWED_CPUS,
) -> list[FloatArr]:
    """Membrane projections.

    Create a $membrane_patch_size^2 array with $membrane_thickness central columns set to 1, other entries set to 0.
    Next compute 30 different rotations of membrane kernel ($theta in [0, 180, step=6 degrees]).
    Convolve each of these kernels with $img to get HxWx30 array, then z-project the array by taking
    the sum, mean, std, median, max and min to get a HxWx6 array out.

    :param img: img arr
    :type img: np.ndarray
    :param membrane_patch_size: size of kernel, defaults to 19
    :type membrane_patch_size: int, optional
    :param membrane_thickness: width of line down the middle, defaults to 1
    :type membrane_thickness: int, optional
    :param num_workers: number of threads, defaults to N_ALLOWED_CPUS
    :type num_workers: int | None, optional
    :return: List of 6 z-projections of membrane convolutions
    :rtype: List[np.ndarray]
    """
    kernel = np.zeros((membrane_patch_size, membrane_patch_size))
    x0 = membrane_patch_size // 2 - membrane_thickness // 2
    x1 = 1 + membrane_patch_size // 2 + membrane_thickness // 2
    kernel[:, x0:x1] = 1

    all_kernels = [np.rint(_rotate_ts(kernel, angle, reshape=False)) for angle in range(0, 180, 6)]
    # map these across threads to speed up (order unimportant)
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        out_angles: list[npt.NDArray[np.float32]] = list(
            ex.map(
                lambda k: convolve(img, k),  # type: ignore
                all_kernels,
            )
        )
    out_angles_np = np.stack(out_angles, axis=0)
    sum_proj = np.sum(out_angles_np, axis=0)
    mean_proj = np.mean(out_angles_np, axis=0)
    std_proj = np.std(out_angles_np, axis=0)
    median_proj = np.median(out_angles_np, axis=0)
    max_proj = np.amax(out_angles_np, axis=0)
    min_proj = np.amin(out_angles_np, axis=0)
    return [mean_proj, max_proj, min_proj, sum_proj, std_proj, median_proj]


# # %% ===================================MANAGER FUNCTIONS===================================


def _singlescale_singlechannel_features(
    img: FloatArr,
    byte_img: npt.NDArray[np.uint8],
    sigma: int,
    config: FeatureConfig,
) -> list[Arr]:
    assert len(img.shape) == 2, f"img shape {img.shape} wrong, should be 2D/singlechannel"
    results: list[Arr] = []
    mult = 0.4 if config.add_weka_sigma_multiplier else 1
    gaussian_filtered = _singlescale_gaussian(img, sigma, mult)
    if config.gaussian_blur:
        results.append(gaussian_filtered)
    if config.sobel_filter:
        results.append(_singlescale_edges(gaussian_filtered))
    if config.hessian_filter:
        hessian_out = _singlescale_hessian(gaussian_filtered, config.add_mod_trace_det_hessian)
        results += hessian_out

    circle_footprint = _make_footprint(int(np.ceil(sigma)))

    if config.mean:
        results.append(_singlescale_mean(byte_img, circle_footprint))
    if config.minimum:
        results.append(_singlescale_minimum(byte_img, circle_footprint))
    if config.maximum:
        results.append(_singlescale_maximum(byte_img, circle_footprint))
    if config.median:
        results.append(_singlescale_median(byte_img, circle_footprint))

    return results


def _zero_scale_filters(
    img: FloatArr,
    sobel_filter: bool = True,
    hessian_filter: bool = True,
    add_mod_trace: bool = True,
) -> list[Arr]:
    """Weka *always* adds the original image, and if computing edgees and/or hessian,
    adds those for sigma=0. This function does that."""
    out_filtered: list[Arr] = [img]
    if sobel_filter:
        edges = _singlescale_edges(img)
        out_filtered.append(edges)
    if hessian_filter:
        hessian = _singlescale_hessian(img, add_mod_trace)
        out_filtered += hessian
    return out_filtered


def _multiscale_singlechannel(
    raw_img: Arr,
    config: FeatureConfig,
    num_workers: int | None = None,
) -> FloatArr:
    byte_img = raw_img.astype(np.uint8)
    converted_img: FloatArr = np.ascontiguousarray(
        img_as_float32(raw_img)  # type: ignore
    )
    features: list[Arr]
    if config.add_zero_scale_features:
        features = _zero_scale_filters(
            converted_img,
            config.sobel_filter,
            config.hessian_filter,
            config.add_mod_trace_det_hessian,
        )
    else:
        features = []

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        out_sigmas = list(
            ex.map(
                lambda sigma: _singlescale_singlechannel_features(  # type: ignore
                    converted_img, byte_img, sigma, config
                ),
                config.sigmas,
            )
        )

    multiscale_features = chain.from_iterable(out_sigmas)
    features += list(multiscale_features)  # type: ignore

    if config.difference_of_gaussians:
        intensities: list[FloatArr] = []
        for i in range(len(config.sigmas)):
            gaussian_blur_at_sigma: FloatArr = out_sigmas[i][0]  # type: ignore
            intensities.append(gaussian_blur_at_sigma)
        dogs = _difference_of_gaussians(intensities)
        features += dogs

    if config.membrane_projections:
        projections = _membrane_projections(
            converted_img,
            config.membrane_patch_size,
            config.membrane_thickness,
            num_workers,
        )
        features += projections

    if config.bilateral:
        bilateral_filtered = _bilateral(byte_img)
        features += bilateral_filtered

    features_np: Arr = np.stack(features, axis=-1)  # type: ignore
    features_np = features_np.astype(np.float32)
    return features_np


def _multiscale_features(
    raw_img: Arr,
    config: FeatureConfig,
    num_workers: int | None = None,
) -> FloatArr:
    out: list[FloatArr] = []
    n_dims = len(raw_img.shape)
    # (H, W)
    if n_dims == 2:
        return _multiscale_singlechannel(raw_img, config, num_workers)
    if n_dims == 4:
        raise Exception(f"Img shape: {raw_img.shape} - 4D data is not supported!")

    channel_idx = np.argmin(raw_img.shape)
    n_ch = raw_img.shape[channel_idx]
    # (1, H, W) or (H, W, 1)
    if n_ch == 1 and channel_idx == 0:
        return _multiscale_singlechannel(raw_img[0], config, num_workers)
    elif n_ch == 1 and channel_idx == -1:
        return _multiscale_singlechannel(raw_img[:, :, 0], config, num_workers)

    # (C, H, W) -> (H, W, C)
    if n_ch == 3 and channel_idx == 0:
        correct_channel_img = np.transpose(raw_img, (1, 2, 0))
    else:
        correct_channel_img = raw_img

    for channel in range(n_ch):
        slice_arr = correct_channel_img[:, :, channel]
        slice_arr = cast(Arr, slice_arr)
        slice_feats = _multiscale_singlechannel(slice_arr, config, num_workers)
        out.append(slice_feats)
    stacked = np.concatenate(out, axis=-1)
    stacked = cast(FloatArr, stacked)
    return stacked


def get_features(
    raw_img: Arr,
    config: FeatureConfig,
    additional_features: Arr | None = None,
    num_workers: int | None = None,
) -> FloatArr:
    """Wrapper to get multiscale features from raw img with config.

    :param raw_img: raw img arr
    :type raw_img: np.ndarray
    :param config: feature config
    :type config: FeatureConfig
    :return: feature arr
    :rtype: np.ndarray
    """
    ih, iw = raw_img.shape[:2]
    image_features = _multiscale_features(raw_img, config, num_workers)

    if additional_features is not None:
        afh, afw = additional_features.shape[:2]
        assert ih == afh and iw == afw, (
            f"Raw img shape {(ih, iw)} and additional features shape {(afh, afw)} do not match in H and W dimensions"
        )
        image_features = np.concatenate([image_features, additional_features], axis=-1)

    return image_features


# # %% ===================================TRAINING===================================

N_VALS_CUTOFF = 20  # if they have more than 20 classes in arr, throw eror


def _read_file_get_arr(path: str) -> Arr:
    if not exists(path):
        raise FileNotFoundError

    file_ext = path.split(".")[-1].lower()
    if file_ext in ("png", "jpg", "jpeg"):
        img = Image.open(path)
        arr = np.array(img)
    elif file_ext in ("tif", "tiff"):
        arr = imread(path)
    else:
        raise Exception(f"filetype '.{file_ext}' not supported!")

    return arr


def _rescale_unique_vals_to_contiguous_labels(
    arr: Arr,
) -> npt.NDArray[np.uint8]:
    # map from (2D) np array, go from unique values -> classes
    unique_vals = sorted(np.unique(arr))
    if len(unique_vals) > N_VALS_CUTOFF:
        raise Exception("Too many unique values in array! Are you sure this is an integer label array? ")

    out = np.zeros_like(arr, dtype=np.uint8)
    for i, val in enumerate(unique_vals):
        out = np.where(arr == val, i, out)
    return out


def _rescale_RGB_to_contiguous_labels(
    arr: npt.NDArray[np.uint8],
) -> npt.NDArray[np.uint8]:
    # convert RGB arrs -> unique ints -> contiguous labels
    new_arr = arr.astype(np.int64)
    R, G, B = new_arr[:, :, 0], new_arr[:, :, 1], new_arr[:, :, 2]
    unique_mapping = R + 255 * G + (255**2) * B
    return _rescale_unique_vals_to_contiguous_labels(unique_mapping)


def load_labels(path: str) -> npt.NDArray[np.uint8]:
    arr = _read_file_get_arr(path).astype(np.uint8)
    is_RGB = len(arr.shape) == 3 and arr.shape[-1] == 3
    if is_RGB:
        return _rescale_RGB_to_contiguous_labels(arr)
    else:
        return _rescale_unique_vals_to_contiguous_labels(arr)


def _get_labelled_training_data_from_stack(feature_stack: Arr, labels: UInt8Arr) -> tuple[Arr, UInt8Arr]:
    h, w, n_feats = feature_stack.shape
    flat_labels = labels.reshape((h * w))
    flat_features = feature_stack.reshape((h * w, n_feats))

    labelled_mask = np.nonzero(flat_labels)

    fit_data = flat_features[labelled_mask[0], :]
    target_data = flat_labels[labelled_mask[0]]

    return fit_data, target_data


def _shuffle_sample_training_data(
    fit: Arr, target: UInt8Arr, shuffle: bool = True, sample_n: int = -1
) -> tuple[Arr, UInt8Arr]:
    n_samples = target.shape[0]
    all_inds = np.arange(0, n_samples, 1)
    if shuffle:
        np.random.shuffle(all_inds)
    if sample_n > 0:
        sample_inds = all_inds[:sample_n]
        return fit[sample_inds], target[sample_inds]
    else:
        return fit[all_inds], target[all_inds]


def get_training_data(
    feature_stack: Arr,
    labels: UInt8Arr,
    shuffle: bool = True,
    sample_n: int = -1,
) -> tuple[Arr, UInt8Arr]:
    """Get training data from feature stack and labelled image.

    :param feature_stack: feature arr
    :type feature_stack: np.ndarray
    :param labels: labelled img arr
    :type labels: np.ndarray
    :param shuffle: whether to shuffle data, defaults to True
    :type shuffle: bool, optional
    :param sample_n: number of samples to return, defaults to -1 (all)
    :type sample_n: int, optional
    :return: fit data and target data
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    fit_data, target_data = _get_labelled_training_data_from_stack(feature_stack, labels)
    fit_data, target_data = _shuffle_sample_training_data(fit_data, target_data, shuffle, sample_n)
    return fit_data, target_data


def get_model(eta: float = 0.3, gamma: float = 0, max_depth: int = 6, class_weight: str = "balanced") -> XGBClassifier:
    return XGBClassifier(eta=eta, gamma=gamma, max_depth=max_depth, class_weight=class_weight)


def fit_model(model: XGBClassifier, fit_data: Arr, target_data: UInt8Arr) -> XGBClassifier:
    model = model.fit(fit_data, target_data)
    return model


def apply(model: XGBClassifier, feature_stack: Arr) -> UInt8Arr:
    h, w, n_feats = feature_stack.shape
    flat_features = feature_stack.reshape((h * w, n_feats))
    pred_flat = model.predict(flat_features)
    pred_flat += 1
    pred_img = pred_flat.reshape((h, w))
    return pred_img


if __name__ == "__main__":
    cfg = FeatureConfig()
    raw_img = np.random.random((300, 300))
    feats = _multiscale_features(raw_img, cfg)
    print(feats.shape)
