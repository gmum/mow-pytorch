from typing import Union
from factories.memory_operator_factory import create_memory_operator
from torch.optim import Adam
import torch
import argparse
from lightning_modules.base_generative_module import BaseGenerativeModule
from factories.autoencoder.autoencoder_params import AutoEncoderParams
from common.noise_creator import NoiseCreator
from factories.autoencoder.cost_function_factory import get_cost_function
from factories.autoencoder.architecture_factory import get_architecture


class AutoEncoderModule(BaseGenerativeModule[AutoEncoderParams]):

    def __init__(self, hparams: AutoEncoderParams):
        super().__init__(hparams)
        self.__encoder, self.__decoder = get_architecture(hparams.dataset, hparams.latent_dim)
        self.__noise_creator = NoiseCreator(hparams.latent_dim)
        self.__cost_function = get_cost_function(hparams.model, self.__noise_creator)
        self.__memory_operator = create_memory_operator(hparams.memory_length)
        self.__lambda = hparams.lambda_val

    def configure_optimizers(self) -> torch.optim.Optimizer:
        hparams = self.get_hparams()
        optimizer = Adam(self.parameters(), lr=hparams.lr)
        return optimizer

    def get_generator(self) -> torch.nn.Module:
        return self.__decoder

    def get_encoder(self) -> torch.nn.Module:
        return self.__encoder

    def get_decoder(self) -> torch.nn.Module:
        return self.__decoder

    def get_noise_dim(self) -> int:
        return self.get_hparams().latent_dim

    def forward(self, batch: torch.Tensor) -> tuple:
        latent = self.__encoder(batch)
        output_images = self.__decoder(latent)
        return latent, output_images

    def training_step(self, batch: torch.Tensor, _) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        _, _, loss = self.calculate_loss(batch, True)
        return loss

    def validation_step(self, batch: torch.Tensor, _) -> tuple[torch.Tensor, torch.Tensor]:
        latent, output_images, _ = self.calculate_loss(batch, False)
        return latent, output_images

    def calculate_loss(self, batch: torch.Tensor, train: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, output_images = self(batch)

        if train:
            eval_latent = self.__memory_operator(latent)
            log_prefix = 'train'
        else:
            eval_latent = latent
            log_prefix = 'val'

        l1, l2 = self.__cost_function(output_images, eval_latent, batch)
        loss = l1 + self.__lambda*l2

        self.log(f'{log_prefix}_recerr', l1, prog_bar=True, on_epoch=True)
        self.log(f'{log_prefix}_normality', l2, prog_bar=True, on_epoch=True)
        self.log(f'{log_prefix}_loss', loss, prog_bar=True, on_epoch=True)

        return latent, output_images, loss

    @ staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--latent_dim', required=True, type=int, help='latent dimension')
        parser.add_argument('--lambda_val', required=False, type=float, default=1.0, help='value of lambda parameter of a cost function')
        parser.add_argument('--load_checkpoint', required=False, type=str, help='whether to select lambda dynamically')
        return parser
