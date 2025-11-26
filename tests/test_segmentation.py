import pytest

import numpy as np
from tifffile import imread, imwrite
from nDTomo.methods.segmentation import (
    Arr,
    UInt8Arr,
    FeatureConfig,
    load_labels,
    get_model,
    fit_model,
    get_features,
    get_training_data,
    apply,
)


def class_avg_mious(prediction: UInt8Arr, ground_truth: UInt8Arr) -> list[float]:
    ious: list[float] = []
    vals = np.unique(ground_truth)
    for v in vals:
        mask_pred = np.where(prediction == v, 1, 0)
        mask_gt = np.where(ground_truth == v, 1, 0)
        overlap = np.logical_and(mask_pred, mask_gt)
        union = np.logical_or(mask_pred, mask_gt)
        iou = float(np.sum(overlap) / np.sum(union))
        ious.append(iou)
    return ious


def class_avg_miou(prediction: UInt8Arr, ground_truth: UInt8Arr) -> float:
    mious = class_avg_mious(prediction, ground_truth)
    mean = np.mean(mious)
    return float(mean)


TEST_IMAGE_SHAPE = (512, 512)


@pytest.fixture()
def image() -> Arr:
    return imread("tests/data/segmentation/0.tif")


@pytest.fixture()
def labels() -> Arr:
    return load_labels("tests/data/segmentation/0_labels.tif")


@pytest.fixture()
def feat_cfg() -> FeatureConfig:
    return FeatureConfig(bilateral=True, median=True, maximum=True, add_weka_sigma_multiplier=False)


@pytest.fixture()
def ground_truth() -> Arr:
    return load_labels("tests/data/segmentation/0_ground_truth.tif") + 1


MIOU_CUTOFF = 0.55


def test_cpu_e2e(
    image: Arr,
    labels: UInt8Arr,
    feat_cfg: FeatureConfig,
    ground_truth: UInt8Arr,
    save: bool = True,
    fname: str = "tests/out/0_seg.tif",
    run_checks: bool = True,
    miou_cutoff: float = MIOU_CUTOFF,
) -> tuple[float, UInt8Arr]:
    features = get_features(image, feat_cfg)
    fit, target = get_training_data(features, labels)
    model = get_model()
    model = fit_model(model, fit, target)
    pred = apply(model, features)
    rh, rw = pred.shape
    fh, fw, _ = features.shape

    miou = class_avg_miou(pred, ground_truth)

    if save:
        imwrite(fname, pred.astype(np.uint8))

    if run_checks:
        assert rh == fh
        assert rw == fw
        print(f"mIoU of segmentation: {miou:.4f}")
        assert miou > miou_cutoff


if __name__ == "__main__":
    pytest.main(args=["-k test_segmentation", "-s"])
