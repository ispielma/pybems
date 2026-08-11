import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")

from pybeams import (  # noqa: E402
    AnnularAperture,
    AntiAperture,
    Axicon,
    CircularAperture,
    ElementLocation,
    IdealLens,
    PropagationResult,
)
from pybeams.plotting import _blocked_segments, plot_propagation  # noqa: E402


class ElementDrawingTests(unittest.TestCase):
    def test_aperture_segments_show_opaque_regions(self):
        self.assertEqual(
            _blocked_segments(CircularAperture(0.4), 1.0),
            ((-1.0, -0.4), (0.4, 1.0)),
        )
        self.assertEqual(
            _blocked_segments(AntiAperture(0.4), 1.0),
            ((-0.4, 0.4),),
        )
        self.assertEqual(
            _blocked_segments(AnnularAperture(0.2, 0.7), 1.0),
            ((-0.2, 0.2), (-1.0, -0.7), (0.7, 1.0)),
        )

    def test_finite_clear_radius_adds_outer_segments(self):
        self.assertEqual(
            _blocked_segments(IdealLens(0.2, clear_radius=0.4), 1.0),
            ((-1.0, -0.4), (0.4, 1.0)),
        )
        self.assertEqual(
            _blocked_segments(Axicon(0.1, 1.45, clear_radius=0.4), 1.0),
            ((-1.0, -0.4), (0.4, 1.0)),
        )
        self.assertEqual(_blocked_segments(IdealLens(0.2), 1.0), ())

    def test_apertures_omit_thin_full_height_field_line(self):
        radius = np.linspace(0.1, 1.0, 10)
        z = np.asarray([0.0, 1.0])
        field = np.ones((radius.size, z.size), dtype=complex)
        result = PropagationResult(
            r=radius,
            z=z,
            field=field,
            elements=(
                ElementLocation(0.25, CircularAperture(0.4)),
                ElementLocation(0.50, AntiAperture(0.3)),
                ElementLocation(0.75, IdealLens(0.2, clear_radius=0.5)),
            ),
        )

        figure, (_, field_axis, _) = plot_propagation(
            result,
            logarithmic=False,
            length_scale=1.0,
            length_unit="m",
        )
        self.assertEqual(len(field_axis.lines), 1)
        self.assertEqual(float(field_axis.lines[0].get_xdata()[0]), 0.75)
        figure.clear()


if __name__ == "__main__":
    unittest.main()
