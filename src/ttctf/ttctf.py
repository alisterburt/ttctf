import einops
import numpy as np
import torch
import typer

from ttctf.estimate_background_2d import estimate_background_2d
from ttctf.estimate_defocus_1d import estimate_defocus_1d
from ttctf.estimate_defocus_2d import estimate_defocus_2d
from ttctf.patch_grid import patch_grid


def ttctf(
    image: np.ndarray,  # (t, h, w) or (h, w)
    pixel_spacing_angstroms: float,
    defocus_model_resolution: tuple[int, int, int],  # (t, h, w)
    frequency_fit_range_angstroms: tuple[float, float],  # (low, high)
    defocus_range_microns: tuple[float, float],  # (low, high)
    voltage_kev: float,
    spherical_aberration_mm: float,
    amplitude_contrast: float,
    patch_sidelength: int = 256
):
    image = torch.tensor(image).float()
    if image.ndim == 2:
        image = einops.rearrange(image, 'h w -> 1 h w')
    t, h, w = image.shape

    # normalise based on stats from central 50% in each dim
    h_low, h_high = int(0.25 * h), int(0.75 * h)
    w_low, w_high = int(0.25 * w), int(0.75 * w)
    image_center = image[:, h_low:h_high, w_low:w_high]
    image_mean = einops.reduce(image_center, 't h w -> 1 1 1', reduction='mean')
    image_std = torch.std(image_center, dim=(-1, -2), keepdim=True)
    image = (image - image_mean) / image_std

    # calculate power spectra from overlapping patches
    patches, patch_centers = patch_grid(
        images=image,
        patch_shape=(1, patch_sidelength, patch_sidelength),
        patch_step=(1, patch_sidelength // 2, patch_sidelength // 2)
    )
    patch_ps = torch.abs(torch.fft.rfftn(patches, dim=(-2, -1))) ** 2

    # estimate defocus in 1D from mean of power spectra to initialise 2D model
    mean_ps = einops.reduce(patch_ps, '... ph pw -> ph pw', reduction='mean')
    initial_defocus_estimate = estimate_defocus_1d(
        power_spectrum=mean_ps,
        image_sidelength=patch_sidelength,
        frequency_fit_range_angstroms=frequency_fit_range_angstroms,
        defocus_range_microns=defocus_range_microns,
        voltage_kev=voltage_kev,
        spherical_aberration_mm=spherical_aberration_mm,
        amplitude_contrast=amplitude_contrast,
        pixel_spacing_angstroms=pixel_spacing_angstroms
    )

    # estimate 2D background and subtract prior to 2D defocus estimation
    background_2d = estimate_background_2d(
        power_spectrum=mean_ps,
        image_sidelength=patch_sidelength,
    )
    patch_ps -= background_2d

    # estimate defocus in 2D with gradient based optimisation
    normalised_patch_positions = patch_centers / torch.tensor([t - 1, h - 1, w - 1]).float().to(patch_ps.device)
    defocus_model = estimate_defocus_2d(
        patch_power_spectra=patch_ps,
        normalised_patch_positions=normalised_patch_positions,
        model_resolution=defocus_model_resolution,
        frequency_fit_range_angstroms=frequency_fit_range_angstroms,
        initial_defocus=initial_defocus_estimate,
        pixel_spacing_angstroms=pixel_spacing_angstroms
    )
    return defocus_model
