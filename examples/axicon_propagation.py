"""Propagate a Gaussian beam through an ideal thin axicon."""

import numpy as np

from pybeams import Axicon, FreeSpace, GaussianBeam, OpticalSystem
from pybeams.plotting import plot_propagation


mm = 1e-3
nm = 1e-9
degrees = np.pi / 180

system = OpticalSystem(
    wavelength=780 * nm,
    max_radius=12.5 * mm,
    grid_size=2048,
)
source = GaussianBeam(waist=2.5 * mm)
elements = [
    Axicon(
        base_angle=5 * degrees,
        refractive_index=1.45,
        clear_radius=12.5 * mm,
    ),
    FreeSpace(distance=200 * mm, dz=1 * mm),
]

result = system.propagate(source, elements)
figure, _ = plot_propagation(result, logarithmic=False)
figure.show()
