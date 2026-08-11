"""Initial fields for axisymmetric propagation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .system import OpticalSystem


ComplexArray = NDArray[np.complex128]


class BeamSource(ABC):
    """An initial radial field sampled on an :class:`OpticalSystem` grid."""

    @abstractmethod
    def field(self, system: OpticalSystem) -> ComplexArray:
        """Return the complex field amplitude on ``system.r``."""


@dataclass(frozen=True)
class GaussianBeam(BeamSource):
    """Gaussian field with ``waist`` equal to its 1/e amplitude radius."""

    waist: float
    amplitude: complex = 1.0
    radius_of_curvature: float = np.inf

    def field(self, system: OpticalSystem) -> ComplexArray:
        if self.waist <= 0:
            raise ValueError("waist must be positive")

        field = np.asarray(
            self.amplitude * np.exp(-(system.r / self.waist) ** 2),
            dtype=np.complex128,
        )
        if np.isfinite(self.radius_of_curvature):
            if self.radius_of_curvature == 0:
                raise ValueError("radius_of_curvature cannot be zero")
            field *= np.exp(
                -0.5j
                * system.k0
                * system.r**2
                / self.radius_of_curvature
            )
        return field


@dataclass(frozen=True)
class TopHatBeam(BeamSource):
    """Uniform field inside ``radius`` and zero outside it."""

    radius: float
    amplitude: complex = 1.0

    def field(self, system: OpticalSystem) -> ComplexArray:
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        return np.asarray(
            self.amplitude * (system.r <= self.radius),
            dtype=np.complex128,
        )
