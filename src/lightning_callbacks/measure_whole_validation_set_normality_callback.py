from typing import Callable
from common.noise_creator import NoiseCreator
from lightning_modules.autoencoder_module import AutoEncoderModule
from pytorch_lightning.trainer.trainer import Trainer
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule
from metrics.cw import cw_normality
from metrics.mmd import mmd_wrapper
from torch import std, mean


class MeasureWholeValidationSetNormalityCallback(Callback):

    def __init__(self, noise_creator: NoiseCreator):
        self.__noise_creator = noise_creator
        self.__latent_evaluators: list[tuple[str, Callable[[torch.Tensor], torch.Tensor]]] = [
            ('cw_normal', cw_normality),
            ('mmd_normal', lambda z: mmd_wrapper(z, self.__noise_creator)),
            ('mean', mean),
            ('std', std),
        ]

    def on_validation_epoch_start(self, _: Trainer, pl_module: LightningModule) -> None:
        self.__encoded_latents = list()

    def on_validation_batch_start(self, _: Trainer, pl_module: AutoEncoderModule, batch: torch.Tensor, ___: int, ____: int) -> None:
        assert self.__encoded_latents is not None
        encoder: torch.nn.Module = pl_module.get_encoder()
        current_latent = encoder(batch.cuda()).detach()
        self.__encoded_latents.append(current_latent)

    def on_validation_epoch_end(self, _: Trainer, pl_module: AutoEncoderModule) -> None:
        assert pl_module.logger
        assert self.__encoded_latents is not None
        full_encoded_latent = torch.cat(self.__encoded_latents)
        max_count = 8192
        for metric_name, latent_evaluator in self.__latent_evaluators:
            metric_values = list()
            for _not_used in range(128):
                sub_sample = full_encoded_latent[torch.randperm(full_encoded_latent.size(0))[:max_count]]
                metric_value = latent_evaluator(sub_sample)
                metric_values.append(metric_value)
            metric_values = torch.stack(metric_values)
            mean_value = metric_values.mean()
            pl_module.log(metric_name, mean_value, prog_bar=True, on_epoch=True)
