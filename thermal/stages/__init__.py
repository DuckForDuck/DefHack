"""Processing modules, one class per step of the pipeline (see docs)."""

from .badpixels import BadPixels
from .clahe import LocalContrast
from .base_detail import BaseDetailSplit
from .compare import Compare
from .decode import Decode
from .hotspot import HotSpotContrast
from .detail import DetailEnhance
from .destripe import Destripe
from .multiscale import MultiScaleDetail
from .naive import NaiveAGC
from .nuc import ResidualNUC
from .preview import Preview, to_preview_uint8
from .quantise import Quantise
from .shutter import ShutterGuard, ShutterHold
from .spatial import SpatialDenoise
from .temporal import TemporalDenoise
from .tonemap import ToneMap
from .tophat import CompactHotBoost
from .upscale import Upscale
from .video import VideoWriter, write_range_test

__all__ = ["BadPixels", "BaseDetailSplit", "Compare", "CompactHotBoost", "Decode", "DetailEnhance", "Destripe", "HotSpotContrast", "LocalContrast",
           "MultiScaleDetail", "NaiveAGC", "Preview", "Quantise",
           "ResidualNUC", "ShutterGuard", "ShutterHold", "SpatialDenoise", "TemporalDenoise", "ToneMap", "Upscale",
           "VideoWriter", "write_range_test",
           "to_preview_uint8"]
