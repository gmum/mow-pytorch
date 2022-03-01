import unittest
import torch
from memory.memory_operator import MemoryOperator


class TestMemoryOperator(unittest.TestCase):

    def test_memory_operator_correctly_appends_size(self):
        # Arrange
        memory_length = 480
        latent_dim = 8
        batch_size = 32
        uut = MemoryOperator(memory_length)

        # Act
        latent = uut(torch.randn((batch_size, latent_dim)))

        # Assert
        self.assertEqual(latent.size(0), 32)
        self.assertEqual(latent.size(1), 8)

    def test_memory_operator_correctly_appends_size_second_time(self):
        # Arrange
        memory_length = 480
        latent_dim = 8
        batch_size = 32
        uut = MemoryOperator(memory_length)

        # Act
        latent = uut(torch.randn((batch_size, latent_dim)))
        latent = uut(torch.randn((batch_size, latent_dim)))

        # Assert
        self.assertEqual(latent.size(0), 64)
        self.assertEqual(latent.size(1), 8)

    def test_memory_operator_correctly_appends_elements(self):
        # Arrange
        memory_length = 480
        latent_dim = 8
        batch_size = 32
        uut = MemoryOperator(memory_length)

        # Act
        latent = uut(torch.zeros((memory_length, latent_dim)))
        latent = uut(torch.ones((batch_size, latent_dim)))

        # Assert
        self.assertEqual(latent.size(0), 512)
        self.assertEqual(latent.count_nonzero(), 32 * 8)

    def test_memory_operator_correctly_appends_elements_second_time(self):
        # Arrange
        memory_length = 480
        latent_dim = 8
        batch_size = 32
        uut = MemoryOperator(memory_length)

        # Act
        latent = uut(torch.zeros((memory_length, latent_dim)))
        latent = uut(torch.ones((batch_size, latent_dim)))
        latent = uut(2 * torch.ones((batch_size, latent_dim)))

        # Assert
        self.assertEqual(latent.count_nonzero(), 2 * batch_size * latent_dim)

    def test_memory_operator_correctly_appends_elements_multiple_times_for_simple_scenario(self):
        # Arrange
        memory_length = 4
        latent_dim = 1
        batch_size = 1
        uut = MemoryOperator(memory_length)
        latent = torch.zeros(memory_length, latent_dim)

        # Act
        uut(latent)
        for i in range(5):
            latent = uut((i+1) * torch.ones((batch_size, latent_dim)))

        # Assert
        self.assertEqual(latent[0], 5)
        self.assertEqual(latent[1], 4)
        self.assertEqual(latent[2], 3)
        self.assertEqual(latent[3], 2)
        self.assertEqual(latent[4], 1)

    def test_memory_operator_correctly_appends_elements_multiple_times_for_simple_scenario_with_forgetting(self):
        # Arrange
        memory_length = 4
        latent_dim = 1
        batch_size = 1
        uut = MemoryOperator(memory_length)
        latent = torch.zeros(memory_length, latent_dim)

        # Act
        uut(latent)
        for i in range(10):
            latent = uut((i+1) * torch.ones((batch_size, latent_dim)))

        # Assert
        self.assertEqual(latent[0], 10)
        self.assertEqual(latent[1], 9)
        self.assertEqual(latent[2], 8)
        self.assertEqual(latent[3], 7)
        self.assertEqual(latent[4], 6)
        self.assertEqual(latent.size(0), 5)
