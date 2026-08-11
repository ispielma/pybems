"""Ideal axisymmetric optical elements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .system import OpticalSystem


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


class OpticalElement(ABC):
    """Base class for an operation on a radial scalar field."""

    @abstractmethod
    def apply(self, field: ComplexArray, system: OpticalSystem) -> ComplexArray:
        """Apply the element and return the resulting radial field."""


class ThinElement(OpticalElement):
    """Base class for ideal elements represented by a transmission function."""

    @abstractmethod
    def transmission(self, system: OpticalSystem) -> ComplexArray:
        """Return the element's complex radial transmission."""

    def apply(self, field: ComplexArray, system: OpticalSystem) -> ComplexArray:
        return np.asarray(field * self.transmission(system), dtype=np.complex128)


@dataclass(frozen=True)
class FreeSpace(OpticalElement):
    """Homogeneous propagation through ``distance`` at a fixed refractive index."""

    distance: float
    refractive_index: float = 1.0
    dz: float = 1e-3

    def apply(self, field: ComplexArray, system: OpticalSystem) -> ComplexArray:
        _, history = self.propagate(field, system)
        return history[:, -1]

    def propagate(
        self,
        field: ComplexArray,
        system: OpticalSystem,
    ) -> tuple[FloatArray, ComplexArray]:
        return system.propagate_free_space(
            field,
            distance=self.distance,
            refractive_index=self.refractive_index,
            dz=self.dz,
        )


@dataclass(frozen=True)
class IdealLens(ThinElement):
    """Ideal thin lens, optionally clipped by a circular clear aperture."""

    focal_length: float
    clear_radius: float = np.inf
    surrounding_index: float = 1.0

    def transmission(self, system: OpticalSystem) -> ComplexArray:
        if self.focal_length == 0:
            raise ValueError("focal_length cannot be zero")
        if self.clear_radius <= 0 or self.surrounding_index <= 0:
            raise ValueError("clear_radius and surrounding_index must be positive")

        phase = np.exp(
            -0.5j
            * system.k0
            * self.surrounding_index
            * system.r**2
            / self.focal_length
        )
        return np.asarray(
            phase * (system.r <= self.clear_radius),
            dtype=np.complex128,
        )


@dataclass(frozen=True)
class Axicon(ThinElement):
    """Ideal thin conical phase element.

    ``base_angle`` is the physical angle between a conical surface and a plane
    normal to the optical axis. The constant center thickness is omitted because
    it contributes only a global phase.
    """

    base_angle: float
    refractive_index: float
    surrounding_index: float = 1.0
    clear_radius: float = np.inf

    def transmission(self, system: OpticalSystem) -> ComplexArray:
        if self.refractive_index <= 0 or self.surrounding_index <= 0:
            raise ValueError("refractive indices must be positive")
        if self.clear_radius <= 0:
            raise ValueError("clear_radius must be positive")

        optical_path_slope = (
            self.refractive_index - self.surrounding_index
        ) * np.tan(self.base_angle)
        phase = np.exp(-1j * system.k0 * optical_path_slope * system.r)
        return np.asarray(
            phase * (system.r <= self.clear_radius),
            dtype=np.complex128,
        )


@dataclass(frozen=True)
class CircularAperture(ThinElement):
    """Transmit the field at radii less than or equal to ``radius``."""

    radius: float

    def transmission(self, system: OpticalSystem) -> ComplexArray:
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        return np.asarray(system.r <= self.radius, dtype=np.complex128)


@dataclass(frozen=True)
class AntiAperture(ThinElement):
    """Block the central region and transmit radii greater than ``radius``."""

    radius: float

    def transmission(self, system: OpticalSystem) -> ComplexArray:
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        return np.asarray(system.r > self.radius, dtype=np.complex128)


@dataclass(frozen=True)
class AnnularAperture(ThinElement):
    """Transmit only between ``inner_radius`` and ``outer_radius``."""

    inner_radius: float
    outer_radius: float

    def transmission(self, system: OpticalSystem) -> ComplexArray:
        if self.inner_radius < 0 or self.outer_radius <= self.inner_radius:
            raise ValueError("require 0 <= inner_radius < outer_radius")
        mask = (system.r >= self.inner_radius) & (system.r <= self.outer_radius)
        return np.asarray(mask, dtype=np.complex128)
