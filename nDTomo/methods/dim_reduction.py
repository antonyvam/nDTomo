import numpy as np
from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans

from sklearn.preprocessing import RobustScaler, minmax_scale, scale, robust_scale

from typing import Literal

DimensionalityReductionMethod = Literal["PCA", "NMF", "KMeans"]
SamplingMethod = Literal[None, "random", "grid"]
PreprocessingMethod = Literal[None, "minmax", "std", "robust"]


def flatten(arr: np.ndarray) -> np.ndarray:
    c = arr.shape[-1]
    return arr.reshape(-1, c)


def sample_data(arr: np.ndarray, method: SamplingMethod, n_samples: int) -> np.ndarray:
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


def apply_preprocessing(flat_arr: np.ndarray, method: PreprocessingMethod) -> np.ndarray:
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
    arr: np.ndarray,
    method: DimensionalityReductionMethod,
    n_components: int,
    preprocessing: PreprocessingMethod,
    sample_method: SamplingMethod = None,
    n_samples: int = 100_000,
    cast_to: np.dtype = np.float32,
) -> object:
    arr = arr.astype(cast_to)

    if method == "PCA":
        model = PCA(n_components=n_components)
    elif method == "NMF":
        model = NMF(n_components=n_components, init="random", random_state=0)
    elif method == "KMeans":
        model = KMeans(n_clusters=n_components, random_state=0)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")

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

    model.fit(sampled)
    return model


def apply_dim_reduction(
    arr: np.ndarray,
    model: PCA | NMF | KMeans,
    preprocessing: PreprocessingMethod,
    cast_to: np.dtype = np.float32,
) -> np.ndarray:
    arr = arr.astype(cast_to)
    flat_arr = flatten(arr)
    scaled = apply_preprocessing(flat_arr, preprocessing)
    transformed = model.transform(scaled)
    transformed_nd = transformed.reshape(arr.shape)
    transformed_nd = transformed_nd.astype(cast_to)
    return transformed_nd
