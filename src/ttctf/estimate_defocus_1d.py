import einops
import torch
from torch_cubic_spline_grids import CubicBSplineGrid1d

from .ctf import calculate_ctf_1d
from .rotational_average import rotational_average_dft_2d
from .utils.dft_utils import spatial_frequency_to_fftfreq


def estimate_defocus_1d(
    power_spectrum: torch.Tensor,
    image_sidelength: int,
    frequency_fit_range_angstroms: tuple[float, float],
    defocus_range_microns: tuple[float, float],
    voltage_kev: float,
    spherical_aberration_mm: float,
    amplitude_contrast: float,
    pixel_spacing_angstroms: float,
) -> torch.Tensor:
    """

    Parameters
    ----------
    power_spectrum: torch.Tensor
        `(h, w)` array containing 2D rfft (no fftshift applied).
    image_sidelength: int
        Sidelength of 2D images prior to rfft calculation.
    frequency_fit_range_angstroms: tuple[float, float]
        `(low, high)` spatial frequency cutoffs for fitting in angstroms.
    defocus_range_microns: tuple[float, float]
        `(low, high)` defoci in microns for initial 1D fit.
    pixel_spacing_angstroms: float
        Isotropic pixel spacing in angstroms.
    Returns
    -------

    """
    # calculate 1d power spectrum
    h, w = image_sidelength, image_sidelength
    rotationally_averaged_power_spectrum, _ = rotational_average_dft_2d(
        power_spectrum,
        image_shape=(h, w),
        rfft=True,
        fftshifted=False,
    )

    # determine subset of 1D values to use based on fit range
    freqs = torch.fft.rfftfreq(h)
    low_ang, high_ang = frequency_fit_range_angstroms
    low_fftfreq = spatial_frequency_to_fftfreq(1 / low_ang, spacing=pixel_spacing_angstroms)
    high_fftfreq = spatial_frequency_to_fftfreq(1 / high_ang, spacing=pixel_spacing_angstroms)
    fit_mask = torch.logical_and(freqs >= low_fftfreq, freqs <= high_fftfreq)
    raps_in_fit_range = rotationally_averaged_power_spectrum[fit_mask]

    # estimate 1D background by fitting a cubic B-spline with 3 control points
    # fit to log(values) for numerical stability
    background_model = CubicBSplineGrid1d(resolution=3)
    background_optimiser = torch.optim.Adam(
        params=background_model.parameters(),
        lr=1
    )
    x = torch.linspace(0, 1, steps=len(raps_in_fit_range))
    y = torch.log(raps_in_fit_range)

    for i in range(200):
        # calculate loss which will be minimised
        prediction = background_model(x).squeeze()
        difference = prediction - y
        mean_squared_error = torch.mean(difference ** 2)

        # backprop, step and zero gradients
        mean_squared_error.backward()
        background_optimiser.step()
        background_optimiser.zero_grad()

    # subtract background model from values
    background = torch.exp(background_model(x).squeeze())
    raps_in_fit_range -= background

    # simulate a set of 1D ctf^2 at different defoci to find best match
    defocus_step = 0.01  # microns
    test_defoci = torch.arange(
        start=defocus_range_microns[0],
        end=defocus_range_microns[1] + defocus_step,
        step=defocus_step,
    )
    ctf2 = calculate_ctf_1d(
        defocus=test_defoci,
        voltage=voltage_kev,
        spherical_aberration=spherical_aberration_mm,
        amplitude_contrast=amplitude_contrast,
        b_factor=0,
        phase_shift=0,
        pixel_size=pixel_spacing_angstroms,
        n_samples=h // 2 + 1,
        oversampling_factor=3,
    ) ** 2

    # fit only in fitting range
    simulated_ctf2_in_fit_range = ctf2[:, fit_mask]

    # normalise simulated values in fitting range
    simulated_ctf2_norms = torch.linalg.norm(simulated_ctf2_in_fit_range, dim=-1, keepdim=True)
    simulated_ctf2_in_fit_range = simulated_ctf2_in_fit_range / simulated_ctf2_norms

    # normalise experimental values in fitting range
    raps_in_fit_range_norm = torch.linalg.norm(raps_in_fit_range)
    normalised_raps_in_fit_range = raps_in_fit_range / raps_in_fit_range_norm

    # calculate zero normalised cross correlation
    zncc = einops.einsum(
        simulated_ctf2_in_fit_range,
        normalised_raps_in_fit_range,
        'b i, i -> b'
    )
    max_correlation_idx = torch.argmax(zncc)
    best_defocus = test_defoci[max_correlation_idx]
    return best_defocus





