from pathlib import Path
from typing import Optional, Annotated

import typer
import torch

from .ttctf import ttctf

cli = typer.Typer(name='ttctf', no_args_is_help=True, add_completion=False)

@cli.command(no_args_is_help=True)
def ttctf_cli(
    input_image: Annotated[Path, typer.Option(show_default=False)],
    pixel_spacing_angstroms: Annotated[Optional[float], typer.Option('--pixel-spacing-angstroms', show_default=False, help="Pixel spacing in Å/px, taken from header if not set")] = None,
    defocus_model_resolution: Annotated[tuple[int, int, int], typer.Option(help='Spatiotemporal resolution of defocus model (x, y, t)')] = (2, 2, 1),
    frequency_fit_range_angstroms: Annotated[tuple[float, float], typer.Option(help='Range of spatial frequencies (low, high) in Å used for fitting')] = (40, 5),
    defocus_range_microns: Annotated[tuple[float, float], typer.Option(help='Range of defoci (low, high) in microns used for initial 1D defocus search')] = (0, 8),
    voltage_kev: Annotated[float, typer.Option(help='Acceleration voltage of microscope')] = 300,
    spherical_aberration_mm: Annotated[float, typer.Option(help='Spherical aberration of imaging system')] = 2.7,
    amplitude_contrast_fraction: Annotated[float, typer.Option(help='Fraction of amplitude contrast in imaging system', min=0, max=1)] = 0.07,
    patch_sidelength_px: Annotated[int, typer.Option(help='Sidelength in pixels of patches used for fitting')] = 512
):
    print(defocus_model_resolution)
