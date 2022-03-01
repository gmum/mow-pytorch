import torch


class NoMemoryOperator:

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        return latent
