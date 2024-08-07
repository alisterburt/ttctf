"""CTF estimation for cryo-EM images with spatiotemporal resolution in PyTorch"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ttctf")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Alister Burt"
__email__ = "alisterburt@gmail.com"

from .ttctf import ttctf
from .cli import cli
