import mrcfile
import torch

from ttctf.ttctf import ttctf

# image parameters
IMAGE_FILE = 'example_data/Position_19_032_36_00_20240228_191749_fractions.mrc'
PIXEL_SIZE = 2.182  # Å/px
VOLTAGE = 300  # kV
SPHERICAL_ABERRATION = 2.7  # mm
AMPLITUDE_CONTRAST = 0.10  # fraction

# model parameters
GRID_RESOLUTION = (1, 2, 2)  # (t, h, w)

# fitting parameters
N_PATCHES_PER_BATCH = 20
PATCH_SIDELENGTH = 512
DEFOCUS_RANGE = (1, 12)  # microns
FITTING_RANGE = (40, 10)  # angstroms


image = mrcfile.read(IMAGE_FILE)

defocus_model = ttctf(
    image=image,
    pixel_spacing_angstroms=PIXEL_SIZE,
    defocus_model_resolution=GRID_RESOLUTION,
    frequency_fit_range_angstroms=FITTING_RANGE,
    defocus_range_microns=DEFOCUS_RANGE,
    voltage_kev=VOLTAGE,
    spherical_aberration_mm=SPHERICAL_ABERRATION,
    amplitude_contrast=AMPLITUDE_CONTRAST
)

print(defocus_model.data)