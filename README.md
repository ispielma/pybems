# pybeams

`pybeams` is a small toolkit for scalar propagation of cylindrically symmetric
optical fields. It uses the quasi-discrete Hankel transform supplied by
[`pyhank`](https://github.com/etfrogers/pyhank), so a radial field is propagated
without constructing a full Cartesian `x-y` grid.

The package currently provides:

- Gaussian and top-hat beam sources.
- Homogeneous free-space propagation, including media with refractive index
  other than one.
- Ideal thin lenses and axicons.
- Circular, central-blocking, and annular apertures.
- A container that owns the wavelength and Hankel grid and propagates a source
  through an ordered list of elements.
- A two-panel propagation plot showing RMS width and full-frame intensity.

All lengths are expressed in SI units.

## Installation

For development:

```bash
python -m pip install -e ".[dev]"
```

## Example

```python
import numpy as np

from pybeams import Axicon, FreeSpace, GaussianBeam, OpticalSystem

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
    Axicon(base_angle=5 * degrees, refractive_index=1.45),
    FreeSpace(distance=200 * mm, dz=1 * mm),
]

result = system.propagate(source, elements)
```

Plotting is deliberately separate from the propagation model:

```python
from pybeams.plotting import plot_propagation

fig, axes = plot_propagation(result)
```

See [`examples/axicon_propagation.py`](examples/axicon_propagation.py) for a
complete runnable example.

## Model scope

The current implementation assumes a monochromatic, scalar, cylindrically
symmetric field. Thin optical elements multiply the radial field by an ideal
complex transmission function. Free-space propagation uses the angular
spectrum propagator on the Hankel-transform grid, including evanescent spatial
frequencies. Interfaces do not yet add Fresnel reflection or refraction losses.
