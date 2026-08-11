"""Optical-system container and propagation results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from pyhank import HankelTransform

from .elements import FreeSpace, OpticalElement
from .sources import BeamSource


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ElementLocation:
    """An optical element and its axial position after propagation."""

    z: float
    element: OpticalElement


@dataclass(frozen=True)
class PropagationResult:
    """Sampled radial fields and derived beam quantities."""

    r: FloatArray
    z: FloatArray
    field: ComplexArray
    elements: tuple[ElementLocation, ...] = ()

    @property
    def intensity(self) -> FloatArray:
        return np.abs(self.field) ** 2

    @property
    def radial_power(self) -> FloatArray:
        return 2 * np.pi * np.trapz(self.intensity * self.r[:, None], self.r, axis=0)

    @property
    def rms_radius(self) -> FloatArray:
        numerator = np.trapz(
            self.intensity * self.r[:, None] ** 3,
            self.r,
            axis=0,
        )
        denominator = np.trapz(
            self.intensity * self.r[:, None],
            self.r,
            axis=0,
        )
        return np.sqrt(numerator / denominator)

    @property
    def rms_x(self) -> FloatArray:
        """RMS width of either Cartesian coordinate for a radial field."""

        return self.rms_radius / np.sqrt(2)

    def nearest_plane(self, z: float) -> tuple[float, ComplexArray]:
        """Return the sampled plane nearest to axial coordinate ``z``."""

        index = int(np.argmin(np.abs(self.z - z)))
        return float(self.z[index]), self.field[:, index].copy()

    def diameter_view(self) -> tuple[FloatArray, ComplexArray]:
        """Mirror the radial field to produce a signed-diameter display."""

        x = np.concatenate((-self.r[::-1], self.r))
        field = np.concatenate((self.field[::-1, :], self.field), axis=0)
        return x, field


class OpticalSystem:
    """Wavelength, radial grid, Hankel transform, and propagation engine."""

    def __init__(
        self,
        wavelength: float,
        max_radius: float,
        grid_size: int,
        *,
        transform_order: int = 0,
        propagation_chunk_size: int = 32,
    ) -> None:
        if wavelength <= 0 or max_radius <= 0:
            raise ValueError("wavelength and max_radius must be positive")
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        if transform_order < 0:
            raise ValueError("transform_order cannot be negative")
        if propagation_chunk_size < 1:
            raise ValueError("propagation_chunk_size must be positive")

        self.wavelength = float(wavelength)
        self.max_radius = float(max_radius)
        self.grid_size = int(grid_size)
        self.transform_order = int(transform_order)
        self.propagation_chunk_size = int(propagation_chunk_size)
        self.transform = HankelTransform(
            order=self.transform_order,
            max_radius=self.max_radius,
            n_points=self.grid_size,
        )

        self.r = np.asarray(self.transform.r, dtype=np.float64)
        self.kr = np.asarray(2 * np.pi * self.transform.v, dtype=np.float64)
        self.k0 = 2 * np.pi / self.wavelength

    def propagate(
        self,
        source: BeamSource,
        elements: Iterable[OpticalElement],
    ) -> PropagationResult:
        """Propagate ``source`` through ``elements`` in their listed order."""

        field = self._validate_field(source.field(self))
        z_now = 0.0
        z_parts = [np.asarray([z_now])]
        field_parts = [field[:, None]]
        locations: list[ElementLocation] = []

        for element in elements:
            if isinstance(element, FreeSpace):
                local_z, local_field = element.propagate(field, self)
                if local_z.size > 1:
                    z_parts.append(z_now + local_z[1:])
                    field_parts.append(local_field[:, 1:])
                z_now += element.distance
                field = local_field[:, -1]
            else:
                field = self._validate_field(element.apply(field, self))
                field_parts[-1][:, -1] = field
            locations.append(ElementLocation(z=z_now, element=element))

        return PropagationResult(
            r=self.r.copy(),
            z=np.concatenate(z_parts),
            field=np.concatenate(field_parts, axis=1),
            elements=tuple(locations),
        )

    def propagate_free_space(
        self,
        field: ComplexArray,
        *,
        distance: float,
        refractive_index: float = 1.0,
        dz: float = 1e-3,
    ) -> tuple[FloatArray, ComplexArray]:
        """Angular-spectrum propagation on the Hankel-transform grid."""

        field = self._validate_field(field)
        if distance < 0:
            raise ValueError("distance cannot be negative")
        if refractive_index <= 0 or dz <= 0:
            raise ValueError("refractive_index and dz must be positive")
        if distance == 0:
            return np.asarray([0.0]), field[:, None]

        number_steps = max(1, int(np.ceil(distance / dz)))
        z = np.linspace(0.0, distance, number_steps + 1)
        spectrum = self.transform.qdht(field)
        wave_number = self.k0 * refractive_index
        kz = np.sqrt((wave_number + 0j) ** 2 - self.kr**2)
        history = np.empty((self.grid_size, z.size), dtype=np.complex128)

        chunk = self.propagation_chunk_size
        for start in range(0, z.size, chunk):
            stop = min(start + chunk, z.size)
            phase = np.exp(1j * kz[:, None] * z[None, start:stop])
            history[:, start:stop] = self.transform.iqdht(
                spectrum[:, None] * phase,
                axis=0,
            )

        return z, history

    def _validate_field(self, field: ComplexArray) -> ComplexArray:
        field = np.asarray(field, dtype=np.complex128)
        if field.shape != (self.grid_size,):
            raise ValueError(
                f"field must have shape ({self.grid_size},), got {field.shape}"
            )
        return field
