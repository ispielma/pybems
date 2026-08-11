import unittest

import numpy as np

from pybeams import (
    AnnularAperture,
    AntiAperture,
    Axicon,
    CircularAperture,
    IdealLens,
    OpticalSystem,
)


class ElementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = OpticalSystem(780e-9, 4e-3, 256)

    def test_aperture_and_anti_aperture_are_complementary(self):
        radius = 1e-3
        aperture = CircularAperture(radius).transmission(self.system)
        anti_aperture = AntiAperture(radius).transmission(self.system)
        np.testing.assert_array_equal(aperture + anti_aperture, 1)

    def test_annular_aperture_has_expected_support(self):
        mask = AnnularAperture(1e-3, 2e-3).transmission(self.system)
        expected = (self.system.r >= 1e-3) & (self.system.r <= 2e-3)
        np.testing.assert_array_equal(mask, expected)

    def test_phase_elements_have_unit_modulus_inside_clear_aperture(self):
        clear_radius = 2e-3
        inside = self.system.r <= clear_radius
        lens = IdealLens(0.2, clear_radius).transmission(self.system)
        axicon = Axicon(np.deg2rad(5), 1.45, clear_radius=clear_radius).transmission(
            self.system
        )
        np.testing.assert_allclose(np.abs(lens[inside]), 1)
        np.testing.assert_allclose(np.abs(axicon[inside]), 1)
        np.testing.assert_array_equal(lens[~inside], 0)
        np.testing.assert_array_equal(axicon[~inside], 0)


if __name__ == "__main__":
    unittest.main()
