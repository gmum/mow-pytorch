from lightning_callbacks.decode_images_callback import DecodeImagesCallback
from common.noise_creator import NoiseCreator
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.base import Callback
from common.args_parser import parse_program_args
from lightning_callbacks.measure_whole_validation_set_normality_callback import MeasureWholeValidationSetNormalityCallback
from lightning_modules.autoencoder_module import AutoEncoderModule
from train_generic import train
from lightning_callbacks.rec_err_evaluator import RecErrEvaluator
from factories.autoencoder.autoencoder_params import AutoEncoderParams


def run():
    parser = parse_program_args()
    parser = AutoEncoderModule.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    hparams: AutoEncoderParams = parser.parse_args()  # type: ignore
    output_dir = f'../results/ae/{hparams.dataset}/{hparams.latent_dim}/{hparams.model}'

    if hasattr(hparams, "load_checkpoint") and hparams.load_checkpoint is not None:
        autoencoder_model = AutoEncoderModule.load_from_checkpoint(hparams.load_checkpoint)
    else:
        autoencoder_model = AutoEncoderModule(hparams)

    noise_creator = NoiseCreator(hparams.latent_dim)

    callbacks: list[Callback] = [
        RecErrEvaluator(),
        MeasureWholeValidationSetNormalityCallback(noise_creator),
        DecodeImagesCallback(64)
    ]

    train(hparams, autoencoder_model, callbacks, output_dir)


if __name__ == '__main__':
    run()
