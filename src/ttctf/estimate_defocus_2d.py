import einops
import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch_cubic_spline_grids import CubicBSplineGrid3d

from ttctf.ctf import calculate_ctf_2d
from ttctf.filters import generate_bandpass_filter
from ttctf.utils.dft_utils import spatial_frequency_to_fftfreq

N_PATCHES_PER_BATCH = 20


def estimate_defocus_2d(
    patch_power_spectra: torch.Tensor,
    normalised_patch_positions: torch.Tensor,
    model_resolution: tuple[int, int, int],
    frequency_fit_range_angstroms: tuple[float, float],
    initial_defocus: float,
    pixel_spacing_angstroms: float,

):
    patch_sidelength = patch_power_spectra.shape[-2]

    # Initialise defocus model as 3D grid with defined resolution at initial defocus
    defocus_grid_data = torch.ones(size=model_resolution) * initial_defocus
    defocus_model = CubicBSplineGrid3d.from_grid_data(defocus_grid_data)

    # bandpass data to fit range
    low_ang, high_ang = frequency_fit_range_angstroms
    low_fftfreq = spatial_frequency_to_fftfreq(1 / low_ang, spacing=pixel_spacing_angstroms)
    high_fftfreq = spatial_frequency_to_fftfreq(1 / high_ang, spacing=pixel_spacing_angstroms)
    filter = generate_bandpass_filter(
        low=low_fftfreq,
        high=high_fftfreq,
        falloff=0,
        image_shape=(patch_sidelength, patch_sidelength),
        rfft=True,
        fftshift=False,
        device=patch_power_spectra.device
    )
    patch_power_spectra *= filter

    # optimise 2d+t defocus model at grid points
    optimiser = torch.optim.Adam(
        params=defocus_model.parameters(),
        lr=0.01,
    )
    scheduler = ExponentialLR(optimiser, gamma=0.9)

    _, ph, pw = normalised_patch_positions.shape[:3]
    for i in range(1000):
        # get random subset of patches and their centers
        patch_idx = np.random.randint(
            low=(0, 0), high=(ph, pw), size=(N_PATCHES_PER_BATCH, 2)
        )
        patch_idx_h, patch_idx_w = einops.rearrange(patch_idx, 'b idx -> idx b')
        subset_patch_ps = patch_power_spectra[:, patch_idx_h, patch_idx_w]
        subset_patch_centers = normalised_patch_positions[:, patch_idx_h, patch_idx_w]

        # get predicted defocus at patch centers
        predicted_patch_defoci = defocus_model(subset_patch_centers)
        predicted_patch_defoci = einops.rearrange(predicted_patch_defoci, '... 1 -> ...')

        # simulate CTFˆ2 at at predicted defocus for each (t, y, x) position
        simulated_ctf2s = calculate_ctf_2d(
            defocus=predicted_patch_defoci,
            voltage=300,
            spherical_aberration=2.7,
            amplitude_contrast=0.10,
            b_factor=0,
            phase_shift=0,
            pixel_size=pixel_spacing_angstroms,
            image_shape=(patch_sidelength, patch_sidelength),
            astigmatism=0,
            astigmatism_angle=0,
            rfft=True,
            fftshift=False,
        ) ** 2  # (t ph pw h w)
        simulated_ctf2s *= filter

        # zero gradients, calculate loss and backpropagate
        optimiser.zero_grad()
        difference = subset_patch_ps - simulated_ctf2s
        mean_squared_error = torch.mean(difference ** 2)
        mean_squared_error.backward()
        optimiser.step()
        if i > 100 and i % 20 == 0:
            scheduler.step()
        if i % 10 == 0:
            print(defocus_model.data)

    return defocus_model
