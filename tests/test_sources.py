import unittest

import numpy as np

from pybeams import GaussianBeam, OpticalSystem, TopHatBeam


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = OpticalSystem(780e-9, 4e-3, 256)

    def test_gaussian_definition(self):
        waist = 0.5e-3
        field = GaussianBeam(waist=waist, amplitude=2).field(self.system)
        expected = 2 * np.exp(-(self.system.r / waist) ** 2)
        np.testing.assert_allclose(field, expected)

    def test_top_hat_definition(self):
        radius = 1e-3
        field = TopHatBeam(radius=radius).field(self.system)
        np.testing.assert_array_equal(field, self.system.r <= radius)

    def test_curved_gaussian_keeps_same_amplitude(self):
        waist = 0.5e-3
        plane = GaussianBeam(waist=waist).field(self.system)
        curved = GaussianBeam(waist=waist, radius_of_curvature=0.3).field(
            self.system
        )
        np.testing.assert_allclose(np.abs(curved), np.abs(plane))


if __name__ == "__main__":
    unittest.main()
