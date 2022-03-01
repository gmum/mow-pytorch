import unittest
import torch
from metrics.cw import cw_normality


class TestCwNormality(unittest.TestCase):

    def test_standard_normal_sample_is_closer_than_nonstandard_normal_sample_greater_mean(self):
        # Arrange
        N, D = (128, 64)
        standard_normal_sample = torch.randn((N, D))
        nonstandard_normal_sample = standard_normal_sample + 0.1

        # Act
        standard_score = cw_normality(standard_normal_sample)
        nonstandard_score = cw_normality(nonstandard_normal_sample)

        # Assert
        self.assertLess(standard_score, nonstandard_score)

    def test_standard_normal_sample_is_closer_than_nonstandard_normal_sample_smaller_mean(self):
        # Arrange
        N, D = (128, 64)
        standard_normal_sample = torch.randn((N, D))
        nonstandard_normal_sample = standard_normal_sample - 0.1

        # Act
        standard_score = cw_normality(standard_normal_sample)
        nonstandard_score = cw_normality(nonstandard_normal_sample)

        # Assert
        self.assertLess(standard_score, nonstandard_score)

    def test_standard_normal_sample_is_closer_than_nonstandard_normal_sample_greater_stddev(self):
        # Arrange
        N, D = (128, 64)
        standard_normal_sample = torch.randn((N, D))
        nonstandard_normal_sample = standard_normal_sample * 1.1

        # Act
        standard_score = cw_normality(standard_normal_sample)
        nonstandard_score = cw_normality(nonstandard_normal_sample)

        # Assert
        self.assertLess(standard_score, nonstandard_score)

    def test_standard_normal_sample_is_closer_than_nonstandard_normal_sample_smaller_stddev(self):
        # Arrange
        N, D = (128, 64)
        standard_normal_sample = torch.randn((N, D))
        nonstandard_normal_sample = standard_normal_sample * 0.9

        # Act
        standard_score = cw_normality(standard_normal_sample)
        nonstandard_score = cw_normality(nonstandard_normal_sample)

        # Assert
        self.assertLess(standard_score, nonstandard_score)

    def test_simple_scenario(self):
        # Arrange
        X = torch.FloatTensor([
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        ])

        cw_dist = cw_normality(X).item()

        # Assert
        self.assertAlmostEqual(0.1049, cw_dist, places=4)

    def test_simple_scenario_invariance(self):
        # Arrange
        X = torch.FloatTensor([
            [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        ])

        cw_dist = cw_normality(X).item()

        # Assert
        self.assertAlmostEqual(0.1049, cw_dist, places=4)

    def test_simple_scenario_v2(self):
        # Arrange
        X = torch.FloatTensor([
            [-0.1, 0.2, -0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8],
            [0.6, 0.6, 0.0, 0.0, 0.6, 0.0, 0.6, 0.0],
            [0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1]
        ])

        cw_dist = cw_normality(X).item()

        # Assert
        self.assertAlmostEqual(0.1027, cw_dist, places=4)
