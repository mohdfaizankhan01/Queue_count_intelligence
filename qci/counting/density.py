"""Backward-compatibility shim: re-exports CSRNetCounter as DensityCounter.

New code should import from ``qci.counting.csrnet`` directly.
"""

from .csrnet import CSRNetCounter as DensityCounter

__all__ = ["DensityCounter"]
