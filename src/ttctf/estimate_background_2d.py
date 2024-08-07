import einops
import torch

import torchvision.transforms.functional as TF

from ttctf.rotational_average import rotational_average_dft_2d


def estimate_background_2d(
    power_spectrum: torch.Tensor,
    image_sidelength: int
):
    raps_2d, _ = rotational_average_dft_2d(
        dft=power_spectrum,
        image_shape=(image_sidelength, image_sidelength),
        rfft=True,
        fftshifted=False,
        return_2d_average=True,
    )
    raps_2d[0, 0] = 0
    raps_2d = einops.rearrange(raps_2d, 'h w -> 1 1 h w')
    bg_estimate_2d = TF.gaussian_blur(raps_2d, kernel_size=25, sigma=10)
    bg_estimate_2d = einops.rearrange(bg_estimate_2d, '1 1 h w -> h w')
    return bg_estimate_2d
