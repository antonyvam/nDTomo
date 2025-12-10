import numpy as np
import torch

from torchnmf.nmf import NMF
from torch_kmeans import KMeans

from nDTomo.methods.dim_reduction import DimensionalityReductionMethod, SamplingMethod, PreprocessingMethod


class TorchPCA(object):
    # from FeatUp (https://github.com/mhamilton723/FeatUp/blob/main/featup/util.py)
    def __init__(self, n_components: int):
        self.n_components = n_components

    def fit(self, X: torch.Tensor) -> "TorchPCA":
        self.mean_ = X.mean(dim=0)
        unbiased = X - self.mean_.unsqueeze(0)
        U, S, V = torch.pca_lowrank(unbiased, q=self.n_components, center=False, niter=4)
        self.components_ = V.T
        self.singular_values_ = S
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        t0 = X - self.mean_.unsqueeze(0)
        projected = t0 @ self.components_.T
        return projected


def flatten(arr: torch.Tensor) -> torch.Tensor:
    c = arr.shape[-1]
    return arr.reshape(-1, c)


def sample_data(arr: torch.Tensor, method: SamplingMethod, n_samples: int) -> torch.Tensor:
    if method is None:
        sampled = flatten(arr)
    elif method == "random":
        flattened = flatten(arr)
        indices = np.random.choice(flattened.shape[0], size=n_samples, replace=False)
        sampled = flattened[indices]
    elif method == "grid":
        is_2d = arr.ndim == 3  # 2D + 1 spectral
        step_size = int(np.prod(arr.shape[:-1]) / n_samples)
        decimated = arr[::step_size, ::step_size] if is_2d else arr[::step_size, ::step_size, ::step_size]
        sampled = flatten(
            decimated,
        )
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    return sampled


def apply_preprocessing(flat_arr: torch.Tensor, method: PreprocessingMethod) -> torch.Tensor:
    if method is None:
        return flat_arr
    elif method == "minmax":
        return minmax_scale(flat_arr, axis=0)
    elif method == "std":
        return scale(flat_arr, axis=0)
    elif method == "robust":
        return robust_scale(flat_arr, axis=0)
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")


def fit_dim_reduction(
    arr: torch.Tensor,
    method: DimensionalityReductionMethod,
    n_components: int,
    preprocessing: PreprocessingMethod,
    sample_method: SamplingMethod = None,
    n_samples: int = 100_000,
    cast_to: torch.dtype = torch.float32,
) -> object:
    arr = arr.to(dtype=cast_to)
    flat_arr = flatten(
        arr,
    )
    scaled = apply_preprocessing(flat_arr, preprocessing)
    # need to reshape to do (potential) spatial sampling
    scaled_nd = scaled.reshape(arr.shape)
    sampled = sample_data(
        scaled_nd,
        sample_method,
        n_samples,
    )

    if method == "PCA":
        model = TorchPCA(n_components=n_components)
    elif method == "NMF":
        model = NMF(sampled.shape, rank=n_components)
    elif method == "KMeans":
        model = KMeans(n_clusters=n_components)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")

    model.fit(sampled)
    return model


def apply_dim_reduction(
    arr: torch.Tensor,
    model: PCA | NMF | KMeans,
    preprocessing: PreprocessingMethod,
    cast_to: torch.dtype = torch.float32,
) -> torch.Tensor:
    arr = arr.to(cast_to)
    flat_arr = flatten(arr)
    scaled = apply_preprocessing(flat_arr, preprocessing)
    transformed = model.transform(scaled)
    transformed_nd = transformed.reshape(arr.shape)
    transformed_nd = transformed_nd.to(cast_to)
    return transformed_nd
