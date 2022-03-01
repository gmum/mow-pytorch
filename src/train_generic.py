from data_modules.dataset_factory import DatasetFactory
from common.noise_creator import NoiseCreator
from lightning_callbacks.generate_sample_images_callback import GenerateSampleImagesCallback
from data_modules.image_dataset_data_module import ImageDatasetDataModule
from lightning_modules.base_generative_module import BaseGenerativeModule
from common.args_parser import BaseArguments
import os
from typing import Union
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.base import Callback
from lightning_callbacks.eval_generative_metrics import EvalGenerativeMetrics


def get_data_module(hparams: BaseArguments) -> ImageDatasetDataModule:
    dataset_factory = DatasetFactory(hparams.dataset, hparams.dataroot)
    data_module = ImageDatasetDataModule(dataset_factory, hparams.batch_size,
                                         hparams.batch_size, hparams.workers)

    return data_module


def train(hparams: BaseArguments, lightning_module: BaseGenerativeModule, callbacks: list[Callback],
          output_dir: str):
    print(f'Using random seed: {pl.seed_everything(hparams.random_seed)}')

    os.makedirs(output_dir, exist_ok=True)
    print('Created output dir: ', output_dir)

    data_module = get_data_module(hparams)

    if isinstance(lightning_module, BaseGenerativeModule):
        noise_creator = NoiseCreator(lightning_module.get_noise_dim())
        if hparams.log_generative_metrics:
            callbacks = [
                EvalGenerativeMetrics(data_module, hparams.enable_kid),
                *callbacks
            ]
        callbacks = [GenerateSampleImagesCallback(noise_creator, 64),
                     *callbacks]

        print('Setting data module')
        lightning_module.set_datamodule(data_module)

    if not hparams.disable_early_stop:
        early_stop = EarlyStopping(
            monitor=hparams.monitor,
            patience=hparams.patience,
            verbose=hparams.verbose,
            mode='min'
        )
        callbacks = [early_stop, *callbacks]

    checkpoint_callback: Union[ModelCheckpoint, None] = None
    if hparams.save_checkpoint:
        checkpoint_callback = ModelCheckpoint(
            verbose=hparams.verbose,
            monitor=hparams.monitor,
            mode='min'
        )
        callbacks = [checkpoint_callback, *callbacks]

    trainer = pl.Trainer.from_argparse_args(hparams,
                                            default_root_dir=output_dir,
                                            callbacks=callbacks)

    trainer.fit(lightning_module, datamodule=data_module)
