import unittest

import numpy as np

from pybeams import CircularAperture, FreeSpace, GaussianBeam, OpticalSystem


class OpticalSystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = OpticalSystem(
            wavelength=780e-9,
            max_radius=4e-3,
            grid_size=512,
            propagation_chunk_size=16,
        )

    def test_hankel_transform_round_trip(self):
        field = GaussianBeam(waist=0.5e-3).field(self.system)
        reconstructed = self.system.transform.iqdht(self.system.transform.qdht(field))
        relative_error = np.linalg.norm(reconstructed - field) / np.linalg.norm(field)
        self.assertLess(relative_error, 1e-11)

    def test_gaussian_free_space_width(self):
        waist = 0.5e-3
        distance = 0.30
        result = self.system.propagate(
            GaussianBeam(waist=waist),
            [FreeSpace(distance=distance, dz=0.01)],
        )
        rayleigh_range = np.pi * waist**2 / self.system.wavelength
        expected_rms_x = 0.5 * waist * np.sqrt(
            1 + (result.z / rayleigh_range) ** 2
        )
        relative_error = np.max(np.abs(result.rms_x / expected_rms_x - 1))
        self.assertLess(relative_error, 2e-3)

    def test_free_space_conserves_radial_power(self):
        result = self.system.propagate(
            GaussianBeam(waist=0.5e-3),
            [FreeSpace(distance=0.30, dz=0.01)],
        )
        relative_variation = np.ptp(result.radial_power) / np.mean(result.radial_power)
        self.assertLess(relative_variation, 2e-3)

    def test_element_locations_and_plane_sampling(self):
        aperture = CircularAperture(1e-3)
        result = self.system.propagate(
            GaussianBeam(waist=0.5e-3),
            [FreeSpace(distance=0.02, dz=0.01), aperture],
        )
        self.assertEqual(result.elements[-1].element, aperture)
        self.assertAlmostEqual(result.elements[-1].z, 0.02)
        sampled_z, sampled_field = result.nearest_plane(0.019)
        self.assertAlmostEqual(sampled_z, 0.02)
        self.assertEqual(sampled_field.shape, (self.system.grid_size,))


if __name__ == "__main__":
    unittest.main()
