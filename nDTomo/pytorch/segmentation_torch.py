import numpy as np

from typing import cast, TYPE_CHECKING

from nDTomo.methods.segmentation import FeatureConfig, _rotate_ts


try:
    import torch
    from torch.nn.functional import conv2d, max_pool2d, avg_pool2d, pad

    torch_imported = True
except ImportError:
    print("GPU dependencies not installed!")
    torch_imported = False
TORCH_AVAILABLE = torch_imported

if TYPE_CHECKING:
    import torch
    from torch.nn.functional import conv2d, max_pool2d, avg_pool2d, pad


# TODO:
# kornia get gaussian 2d, median filter, bilateral ?
# training code


def prepare_for_gpu(arr: np.ndarray, device: str = "cuda:0", dtype: "torch.dtype" = torch.float32) -> "torch.Tensor":
    ndims = len(arr.shape)
    if ndims == 2:
        arr = np.expand_dims(arr, (0, 1))  # (H, W) -> (1, 1, H, W)
    else:
        channel_idx = np.argmin(arr.shape)
        if channel_idx == ndims - 1:  # (H, W, C) -> (C, H, W)
            arr = np.transpose(arr, (-1, 0, 1))
        arr = np.expand_dims(arr, (0))  # (C, H, W) -> (1, 1, H, W)
    tensor = torch.tensor(arr, device=device, dtype=dtype)
    return tensor


# def concat_feats(arr1: Arrlike, arr2: Arrlike) -> Arrlike:
#     # (optionally) cast to tensors and concatenate arrays
#     if isinstance(arr1, torch.Tensor) and isinstance(arr2, torch.Tensor):
#         res = torch.concatenate((arr1, arr2), dim=-1)
#     elif isinstance(arr1, torch.Tensor) and not isinstance(arr2, torch.Tensor):
#         tensor_2 = torch.tensor(arr2, dtype=arr1.dtype, device=arr1.device)
#         res = torch.concatenate((arr1, tensor_2), dim=-1)
#     elif isinstance(arr2, torch.Tensor) and not isinstance(arr1, torch.Tensor):
#         tensor_1 = torch.tensor(arr1, dtype=arr2.dtype, device=arr2.device)
#         res = torch.concatenate((tensor_1, arr2), dim=-1)
#     elif isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray):
#         res = np.concatenate((arr1, arr2), axis=-1)
#     else:
#         raise Exception(f"Invalid feat types: {type(arr1)} + {type(arr2)}")

#     return cast(Arrlike, res)


# %% ===================================KORNIA FILTERS===================================
"""
The following block is taken from kornia.filters.kernels & kornia.filters

kornia: https://github.com/kornia/kornia
Apache 2.0
"""


def _unpack_2d_ks(kernel_size: tuple[int, int] | int) -> tuple[int, int]:
    if isinstance(kernel_size, int):
        ky = kx = kernel_size
    else:
        ky, kx = kernel_size

    ky = int(ky)
    kx = int(kx)

    return (ky, kx)


def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
    device, dtype = None, None
    if isinstance(sigma, torch.Tensor):
        device, dtype = sigma.device, sigma.dtype
    x = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp(-x.pow(2.0) / (2 * sigma**2))
    return gauss / gauss.sum()


def _get_gaussian_kernel1d(kernel_size: int, sigma: float, force_even: bool = False) -> torch.Tensor:
    r"""Function that returns Gaussian filter coefficients.

    Args:
        kernel_size: filter size. It should be odd and positive.
        sigma: gaussian standard deviation.
        force_even: overrides requirement for odd kernel size.

    Returns:
        1D tensor with gaussian filter coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size})`

    Examples:

        >>> get_gaussian_kernel1d(3, 2.5)
        tensor([0.3243, 0.3513, 0.3243])

        >>> get_gaussian_kernel1d(5, 1.5)
        tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
    """
    if not isinstance(kernel_size, int) or ((kernel_size % 2 == 0) and not force_even) or (kernel_size <= 0):
        raise TypeError("kernel_size must be an odd positive integer. Got {}".format(kernel_size))
    window_1d: torch.Tensor = _gaussian(kernel_size, sigma)
    return window_1d


def _get_gaussian_kernel2d(
    kernel_size: tuple[int, int],
    sigma: tuple[float, float],
    force_even: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size: filter sizes in the x and y direction.
         Sizes should be odd and positive.
        sigma: gaussian standard deviation in the x and y
         direction.
        force_even: overrides requirement for odd kernel size.

    Returns:
        2D tensor with gaussian filter matrix coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

    Examples:
        >>> get_gaussian_kernel2d((3, 3), (1.5, 1.5))
        tensor([[0.0947, 0.1183, 0.0947],
                [0.1183, 0.1478, 0.1183],
                [0.0947, 0.1183, 0.0947]])
        >>> get_gaussian_kernel2d((3, 5), (1.5, 1.5))
        tensor([[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
                [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
                [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]])
    """
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError(f"kernel_size must be a tuple of length two. Got {kernel_size}")
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError(f"sigma must be a tuple of length two. Got {sigma}")
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = _get_gaussian_kernel1d(ksize_x, sigma_x, force_even)
    kernel_y: torch.Tensor = _get_gaussian_kernel1d(ksize_y, sigma_y, force_even)
    kernel_2d: torch.Tensor = torch.matmul(kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    kernel_2d = kernel_2d.to(device=device, dtype=dtype)
    return kernel_2d


def _compute_zero_padding(kernel_size: tuple[int, int]) -> tuple[int, int]:
    r"""Utility function that computes zero padding tuple."""
    computed: list[int] = [(k - 1) // 2 for k in kernel_size]
    return computed[0], computed[1]


def _median_blur(input: torch.Tensor, kernel_size: tuple[int, int]) -> torch.Tensor:
    r"""
    Blur an image using the median filter.

    .. image:: _static/img/median_blur.png

    Args:
        input: the input image with shape :math:`(B,C,H,W)`.
        kernel_size: the blurring kernel size.

    Returns:
        the blurred input tensor with shape :math:`(B,C,H,W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       filtering_operators.html>`__.

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> output = median_blur(input, (3, 3))
        >>> output.shape
        torch.Size([2, 4, 5, 7])
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")

    padding: tuple[int, int] = _compute_zero_padding(kernel_size)

    # prepare kernel
    kernel: torch.Tensor = _get_binary_kernel2d(kernel_size).to(input)
    b, c, h, w = input.shape

    # map the local window to single vector
    features: torch.Tensor = conv2d(input.reshape(b * c, 1, h, w), kernel, padding=padding, stride=1)
    features = features.view(b, c, -1, h, w)  # BxCx(K_h * K_w)xHxW

    # compute the median along the feature axis
    median: torch.Tensor = torch.median(features, dim=2)[0]

    return median


def _bilateral_blur(
    input: torch.Tensor,
    kernel_size: tuple[int, int] | int,
    sigma_color: float | torch.Tensor,
    sigma_space: tuple[float, float] | torch.Tensor,
    border_type: str = "reflect",
    color_distance_type: str = "l1",
) -> torch.Tensor:
    "Single implementation for both Bilateral Filter and Joint Bilateral Filter"

    ky, kx = _unpack_2d_ks(kernel_size)
    pad_y, pad_x = _compute_zero_padding((ky, kx))

    padded_input = pad(input, (pad_x, pad_x, pad_y, pad_y), mode=border_type)
    unfolded_input = padded_input.unfold(2, ky, 1).unfold(3, kx, 1).flatten(-2)  # (B, C, H, W, Ky x Kx)

    guidance = input
    unfolded_guidance = unfolded_input

    diff = unfolded_guidance - guidance.unsqueeze(-1)
    if color_distance_type == "l1":
        color_distance_sq = diff.abs().sum(1, keepdim=True).square()
    elif color_distance_type == "l2":
        color_distance_sq = diff.square().sum(1, keepdim=True)
    else:
        raise ValueError("color_distance_type only acceps l1 or l2")
    color_kernel = (-0.5 / sigma_color**2 * color_distance_sq).exp()  # (B, 1, H, W, Ky x Kx)

    space_kernel = _get_gaussian_kernel2d((ky, kx), sigma_space, device=input.device, dtype=input.dtype)
    space_kernel = space_kernel.view(-1, 1, 1, 1, kx * ky)

    kernel = space_kernel * color_kernel
    out = (unfolded_input * kernel).sum(-1) / kernel.sum(-1)
    return out


# %% ===================================SINGLESCALE FEATURES===================================


def _get_multiscale_gaussian_kernel(
    device: torch.device,
    dtype: torch.dtype,
    sigmas: tuple[float, ...],
    n_channels: int,
    mult: float = 1.0,
) -> "torch.Tensor":
    # get kernel of shape (N_s, max_k, max_k) where max_k is largest (truncated) gaussian kernel
    N = len(sigmas)
    max_s = max(sigmas)
    max_k = 4 * int(max_s * mult) + 1
    filters = torch.zeros((N, 1, max_k, max_k), dtype=dtype, device=device, requires_grad=False)
    for i, sigma in enumerate(sigmas):
        filters[i, :, :, :] = _get_gaussian_kernel2d(
            (max_k, max_k), (sigma * mult, sigma * mult), device=device, dtype=dtype
        )
    filters = torch.tile(filters, (n_channels, 1, 1, 1))
    return filters


def _get_sobel_kernel(device: torch.device, dtype: torch.dtype, n_channels: int) -> "torch.Tensor":
    g_y = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        dtype=dtype,
        device=device,
        requires_grad=False,
    )
    g_x = torch.tensor(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        dtype=dtype,
        device=device,
        requires_grad=False,
    )

    stacked = torch.stack((g_x, g_y))
    filters = stacked.unsqueeze(1)
    filters = torch.tile(filters, (n_channels, 1, 1, 1))
    return filters


def _reflect_padded(func):  # type: ignore
    def wrapper(
        tensor: torch.Tensor,
        sigma_or_kernel: int | torch.Tensor,
        *args,  # type: ignore
        **kwargs,  # type: ignore
    ) -> "torch.Tensor":
        if isinstance(sigma_or_kernel, torch.Tensor):
            _, _, kh, kw = sigma_or_kernel.shape
        else:
            kh, kw = 2 * sigma_or_kernel + 1, 2 * sigma_or_kernel + 1
        padded = pad(tensor, (kw // 2, kw // 2, kh // 2, kh // 2), mode="reflect")
        return func(padded, sigma_or_kernel, *args, **kwargs)  # type: ignore

    return wrapper


@_reflect_padded
def _convolve(img: torch.Tensor, kernel: torch.Tensor, norm: bool = False) -> "torch.Tensor":
    _, in_ch, _, _ = img.shape
    if norm:
        summand = torch.sum(torch.abs(kernel), dim=(2, 3), keepdim=True)
        kernel_norm = kernel / summand
    else:
        kernel_norm = kernel

    convolved = conv2d(img, kernel_norm, stride=1, groups=in_ch)
    return convolved


def _get_gradient_mag(edges: "torch.Tensor") -> "torch.Tensor":
    g_x = edges[0:1, 0::2]
    g_y = edges[0:1, 1::2]
    return torch.sqrt((g_x**2 + g_y**2))


def _singescale_hessian(
    dx_dy: "torch.Tensor", sobel_kernel: "torch.Tensor", return_full: bool = True
) -> "torch.Tensor":
    """_summary_

    :param dx_dy: (B, 2, H, W) first derivatives from sobel
    :type dx_dy: torch.Tensor
    :param sobel_kernel: (2, 1, 3, 3) sobel kernel
    :type sobel_kernel: torch.Tensor
    :param return_full: return mod, det and trace as well as eigs, defaults to True
    :type return_full: bool, optional
    :return: either
    :rtype: torch.Tensor
    """
    second_deriv = _convolve(dx_dy, sobel_kernel, True)

    a: "torch.Tensor" = second_deriv[0:1, 0::4]
    b: "torch.Tensor" = second_deriv[0:1, 1::4]
    d: "torch.Tensor" = second_deriv[0:1, 3::4]

    mod = torch.sqrt(a**2 + b**2 + d**2)
    trace = a + d
    det = a * d - b**2

    eig1 = trace + torch.sqrt((4 * b**2 + (a - d) ** 2))
    eig2 = trace - torch.sqrt((4 * b**2 + (a - d) ** 2))

    to_stack: tuple[torch.Tensor, ...]
    if return_full:
        to_stack = (eig1 / 2.0, eig2 / 2.0, mod, trace, det)
    else:
        to_stack = (eig1 / 2.0, eig2 / 2.0)
    out = torch.cat(to_stack, dim=1)
    return out


@_reflect_padded
def _singlescale_mean(img: torch.Tensor, sigma: int) -> "torch.Tensor":
    k = 2 * sigma + 1
    out = avg_pool2d(img, k, 1, ceil_mode=True)
    return out


@_reflect_padded
def _singlescale_maximum(img: torch.Tensor, sigma: int) -> "torch.Tensor":
    k = 2 * sigma + 1
    out = max_pool2d(img, k, 1, ceil_mode=True)
    return out


@_reflect_padded
def _singlescale_minimum(img: torch.Tensor, sigma: int) -> "torch.Tensor":
    k = 2 * sigma + 1
    out = -max_pool2d(-img, k, 1, ceil_mode=True)
    return out


def _get_binary_kernel2d(window_size: tuple[int, int]) -> torch.Tensor:
    r"""Create a binary kernel to extract the patches. If the window size
    is HxW will create a (H*W)xHxW kernel.
    """
    window_range: int = window_size[0] * window_size[1]
    kernel: torch.Tensor = torch.zeros(window_range, window_range)
    for i in range(window_range):
        kernel[i, i] += 1.0
    return kernel.view(window_range, 1, window_size[0], window_size[1])


def _singlescale_median(img: torch.Tensor, sigma: int) -> "torch.Tensor":
    k = 2 * sigma + 1
    return _median_blur(img, (k, k))


# %% ===================================SCALE-FREE FEATURES===================================
def _bilateral(img: "torch.Tensor") -> "torch.Tensor":
    bilaterals: list[torch.Tensor] = []
    for spatial_radius in (3, 5):
        for value_range in (50, 100):  # check your pixels are [0, 255]
            k = 2 * spatial_radius + 1
            filtered: "torch.Tensor" = _bilateral_blur(
                img,
                k,
                sigma_color=value_range / 255.0,
                sigma_space=(spatial_radius, spatial_radius),
            )
            bilaterals.append(filtered)
    return torch.cat(bilaterals, dim=1)


def _difference_of_gaussians(gaussian_blurs: torch.Tensor, N_sigmas: int) -> "torch.Tensor":
    diff_list: list[torch.Tensor] = []
    for i in range(N_sigmas):
        sigma_1 = gaussian_blurs[0:1, i::N_sigmas]
        for j in range(i):
            sigma_2 = gaussian_blurs[0:1, j::N_sigmas]
            diff_list.append(sigma_2 - sigma_1)
    dogs = torch.cat(diff_list, dim=1)
    return dogs


def _get_membrane_proj_kernel(
    device: torch.device,
    dtype: torch.dtype,
    n_channels: int,
    membrane_patch_size: int = 17,
    membrane_thickness: int = 1,
    angle_increment_deg: int = 6,
) -> "torch.Tensor":
    kernel = np.zeros((membrane_patch_size, membrane_patch_size))
    x0 = membrane_patch_size // 2 - membrane_thickness // 2
    x1 = 1 + membrane_patch_size // 2 + membrane_thickness // 2
    kernel[:, x0:x1] = 1

    all_kernels = [np.rint(_rotate_ts(kernel, angle, reshape=False)) for angle in range(0, 180, angle_increment_deg)]
    kernel_np = np.stack(all_kernels)
    kernel_torch = torch.tensor(kernel_np, device=device, dtype=dtype, requires_grad=False)
    filters = kernel_torch.unsqueeze(1)
    filters = torch.tile(filters, (n_channels, 1, 1, 1))

    return filters


def _membrane_projections(img: torch.Tensor, kernel: torch.Tensor, N_ch: int) -> "torch.Tensor":
    # TODO: most other gpu filters work with N-channel imgs s.t slicing the output stack
    # by every N you'll get the stack for the Nth channel - this does not work like that
    projs = _convolve(img, kernel, False)

    projs_per_ch = projs.shape[1] // N_ch

    projections_out: list[torch.Tensor] = []
    for i in range(N_ch):
        sum_proj = torch.sum(projs[0:1, i * projs_per_ch : (i + 1) * projs_per_ch], dim=1)
        mean_proj = torch.mean(projs[0:1, i * projs_per_ch : (i + 1) * projs_per_ch], dim=1)
        std_proj = torch.std(projs[0:1, i * projs_per_ch : (i + 1) * projs_per_ch], dim=1)
        median_proj, _ = torch.median(projs[0:1, i * projs_per_ch : (i + 1) * projs_per_ch], dim=1)
        max_proj = torch.amax(projs[0:1, i * projs_per_ch : (i + 1) * projs_per_ch], dim=1)
        min_proj = torch.amin(projs[0:1, i * projs_per_ch : (i + 1) * projs_per_ch], dim=1)
        projections_out += [
            mean_proj,
            max_proj,
            min_proj,
            sum_proj,
            std_proj,
            median_proj,
        ]

    return torch.stack(projections_out, dim=1)


def _zero_scale_filters(
    img: torch.Tensor,
    sobel_kernel: torch.Tensor,
    sobel_squared_kernel: torch.Tensor,
    sobel_filter: bool = True,
    hessian_filter: bool = True,
    add_mod_trace: bool = True,
) -> list[torch.Tensor]:
    """Weka *always* adds the original image, and if computing edgees and/or hessian,
    adds those for sigma=0. This function does that."""
    out_filtered: list[torch.Tensor] = [img]
    edges = _convolve(img, sobel_kernel, True)
    if sobel_filter:
        out_filtered.append(_get_gradient_mag(edges))
    if hessian_filter:
        hessian = _singescale_hessian(edges, sobel_squared_kernel, add_mod_trace)
        out_filtered.append(hessian)
    return out_filtered


@torch.no_grad()
def _multiscale_features_gpu(
    raw_img: torch.Tensor,
    config: FeatureConfig,
    reshape_squeeze: bool = True,
) -> "torch.Tensor":
    dtype = raw_img.dtype
    _, C, _, _ = raw_img.shape
    amax = torch.amax(raw_img)
    converted_img = (raw_img * (1 / amax)).to(dtype)

    device = raw_img.device
    mult = 0.4 if config.add_weka_sigma_multiplier else 1
    gauss_kernel = _get_multiscale_gaussian_kernel(device, dtype, config.sigmas, C, mult)
    sobel_kernel = _get_sobel_kernel(device, dtype, C)
    sobel_squared = _get_sobel_kernel(device, dtype, 2 * C)

    membrane_kernel = _get_membrane_proj_kernel(device, dtype, C, config.membrane_patch_size, config.membrane_thickness)

    gaussian_blurs = _convolve(converted_img, gauss_kernel, norm=False)

    features: list[torch.Tensor]
    if config.add_zero_scale_features:
        features = _zero_scale_filters(
            converted_img,
            sobel_kernel,
            sobel_squared,
            config.sobel_filter,
            config.hessian_filter,
            config.add_mod_trace_det_hessian,
        )
    else:
        features = []

    N_sigmas = len(config.sigmas)
    for i, sigma in enumerate(config.sigmas):
        s = int(sigma)
        blurred = gaussian_blurs[0:1, i::N_sigmas]
        edges = _convolve(blurred, sobel_kernel, True)
        if config.gaussian_blur:
            features.append(blurred)
        if config.sobel_filter:
            features.append(_get_gradient_mag(edges))
        if config.hessian_filter:
            hess = _singescale_hessian(edges, sobel_squared, config.add_mod_trace_det_hessian)
            features.append(hess)

        if config.mean:
            features.append(_singlescale_mean(raw_img, s))
        if config.minimum:
            features.append(_singlescale_minimum(raw_img, s))
        if config.maximum:
            features.append(_singlescale_maximum(raw_img, s))
        if config.median:
            features.append(_singlescale_median(raw_img, s))

    if config.difference_of_gaussians:
        features.append(_difference_of_gaussians(gaussian_blurs, N_sigmas))
    if config.membrane_projections:
        projections = _membrane_projections(converted_img, membrane_kernel, C)
        features.append(projections)
    if config.bilateral:
        features.append(_bilateral(raw_img))

    features_out = torch.cat(features, dim=1)

    if reshape_squeeze:
        features_out = torch.squeeze(features_out, 0)
        features_out = torch.permute(features_out, (1, 2, 0))
    return features_out


def get_features(
    raw_img: torch.Tensor,
    config: FeatureConfig,
    additional_features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Wrapper to get multiscale features from raw img with config.

    :param raw_img: raw img arr
    :type raw_img: np.ndarray
    :param config: feature config
    :type config: FeatureConfig
    :return: feature arr
    :rtype: np.ndarray
    """
    ih, iw = raw_img.shape[:2]
    image_features = _multiscale_features_gpu(raw_img, config)

    if additional_features is not None:
        afh, afw = additional_features.shape[:2]
        assert ih == afh and iw == afw, (
            f"Raw img shape {(ih, iw)} and additional features shape {(afh, afw)} do not match in H and W dimensions"
        )
        image_features = torch.cat([image_features, additional_features], dim=-1)

    return image_features


if __name__ == "__main__":
    cfg = FeatureConfig(bilateral=True, median=True, maximum=True)
    img = torch.rand(1, 1, 64, 64)
    feats = _multiscale_features_gpu(img, cfg)
    print(feats.shape)
