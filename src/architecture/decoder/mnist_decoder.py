import torch
import torch.nn as nn
from modules.linear_block_with_batch_norm import LinearBlockWithNormalization
from modules.view import View


class Decoder(nn.Module):
    def __init__(self, latent_size: int):
        super().__init__()
        neurons_count = 256
        self.__sequential_blocks = [
            LinearBlockWithNormalization(latent_size, neurons_count),
            LinearBlockWithNormalization(neurons_count, neurons_count),
            LinearBlockWithNormalization(neurons_count, neurons_count),
            nn.Linear(neurons_count, 28 * 28),
            nn.Sigmoid(),
            View(-1, 1, 28, 28)
        ]
        self.main = nn.Sequential(*self.__sequential_blocks)

    def forward(self, input_latent: torch.Tensor):
        decoded_images = self.main(input_latent)
        assert decoded_images.size(1) == 1 and decoded_images.size(2) == 28 and decoded_images.size(3) == 28
        return decoded_images
