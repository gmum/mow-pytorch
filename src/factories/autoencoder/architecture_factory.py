from torch.nn import Module
from architecture.encoder.mnist_encoder import Encoder as MnistEncoder
from architecture.decoder.mnist_decoder import Decoder as MnistDecoder
from architecture.encoder.celeba_wae_encoder import Encoder as CelebaWaeEncoder
from architecture.decoder.celeba_wae_decoder import Decoder as CelebaWaeDecoder
from architecture.encoder.svhn_encoder import Encoder as SvhnEncoder
from architecture.decoder.svhn_decoder import Decoder as SvhnDecoder


def get_architecture(identifier: str, z_dim: int) -> tuple[Module, Module]:
    if identifier == 'mnist' or identifier == 'fmnist' or identifier == 'kmnist':
        return MnistEncoder(z_dim), MnistDecoder(z_dim)

    if identifier == 'celeba':
        return CelebaWaeEncoder(z_dim), CelebaWaeDecoder(z_dim)

    if identifier == 'svhn' or identifier == 'cifar10':
        return SvhnEncoder(z_dim), SvhnDecoder(z_dim)

    raise ValueError("Unknown architecture")
