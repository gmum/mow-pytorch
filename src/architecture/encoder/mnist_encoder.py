from pyrsistent import v
import torch
import torch.nn as nn

from modules.linear_block_with_batch_norm import LinearBlockWithNormalization


class Encoder(nn.Module):
    def __init__(self, latent_size: int):
        super().__init__()
        neurons_count = 256
        self.__sequential_blocks = [
            nn.Flatten(start_dim=1),
            LinearBlockWithNormalization(28*28, neurons_count),
            LinearBlockWithNormalization(neurons_count, neurons_count),
            LinearBlockWithNormalization(neurons_count, neurons_count),
            nn.Linear(neurons_count, latent_size)
        ]
        self.main = nn.Sequential(*self.__sequential_blocks)

    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        assert input_images.size(1) == 1 and input_images.size(2) == 28 and input_images.size(3) == 28
        encoded_latent = self.main(input_images)
        return encoded_latent
