from typing import Generic, TypeVar, Union
from common.args_parser import BaseArguments
import torch
import pytorch_lightning as pl
import abc

from data_modules.image_dataset_data_module import ImageDatasetDataModule

T = TypeVar('T', bound=BaseArguments)


class BaseGenerativeModule(pl.LightningModule, Generic[T], metaclass=abc.ABCMeta):

    def __init__(self, hparams: T):
        super().__init__()
        self.__hparams = hparams
        self.datamodule: Union[ImageDatasetDataModule, None] = None
        self.save_hyperparameters()

    def get_hparams(self) -> T:
        return self.__hparams

    @abc.abstractmethod
    def get_generator(self) -> torch.nn.Module:
        pass

    @abc.abstractmethod
    def get_noise_dim(self) -> int:
        pass

    def set_datamodule(self, datamodule: ImageDatasetDataModule) -> None:
        self.datamodule = datamodule
