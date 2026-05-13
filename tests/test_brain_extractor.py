"""Smoke tests for the BrainExtractor pipeline.

These exercise the full surface-deformation loop on a synthetic spherical
"brain" so failures point at integration regressions, not numerical drift in
any single helper.
"""

import nibabel as nib
import numpy as np
import pytest

from brainextractor import BrainExtractor


@pytest.fixture(scope="module")
def synthetic_brain_img():
    """A 64^3 volume with a bright sphere on a dark background."""
    shape = (64, 64, 64)
    center = np.array([32, 32, 32])
    radius = 18

    ii, jj, kk = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
    dist_sq = (ii - center[0]) ** 2 + (jj - center[1]) ** 2 + (kk - center[2]) ** 2
    data = np.where(dist_sq <= radius**2, 1000.0, 50.0).astype(np.float32)

    affine = np.eye(4)
    return nib.Nifti1Image(data, affine), center, radius


def test_initialization_finds_center_of_mass(synthetic_brain_img):
    img, center, _ = synthetic_brain_img
    bet = BrainExtractor(img=img)
    np.testing.assert_allclose(bet.c, center, atol=1.0)


def test_initialization_estimates_reasonable_radius(synthetic_brain_img):
    img, _, radius = synthetic_brain_img
    bet = BrainExtractor(img=img)
    # bet.r is the half-radius estimate; 2 * bet.r should be close to the true radius
    assert bet.r > 0
    assert 2 * bet.r == pytest.approx(radius, rel=0.5)


def test_run_short_iteration_produces_mask(synthetic_brain_img):
    img, _, _ = synthetic_brain_img
    bet = BrainExtractor(img=img)
    bet.run(iterations=5)
    mask = bet.compute_mask()
    assert mask.shape == img.shape
    assert mask.sum() > 0


def test_save_mask_writes_nifti(tmp_path, synthetic_brain_img):
    img, _, _ = synthetic_brain_img
    bet = BrainExtractor(img=img)
    bet.run(iterations=5)

    out_path = tmp_path / "mask.nii.gz"
    bet.save_mask(str(out_path))
    assert out_path.exists()

    loaded = nib.load(str(out_path))
    assert loaded.shape == img.shape
