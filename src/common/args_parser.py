import argparse


class BaseArguments(argparse.Namespace):
    model: str
    dataset: str
    dataroot: str
    workers: int
    batch_size: int
    memory_length: int
    lr: float
    monitor: str
    random_seed: int
    offline_mode: bool
    patience: int
    verbose: bool
    save_checkpoint: bool
    extra_tag: str
    enable_kid: bool
    disable_early_stop: bool
    log_generative_metrics: bool


def parse_program_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', required=True, help='mnist | fmnist | celeba')
    parser.add_argument('--dataroot', default='../data', help='path to dataset')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--memory_length', required=False, type=int, default=0, help='memory length')
    parser.add_argument('--lr', type=float, default=0.001, help='optimizers learning rate value')
    parser.add_argument('--monitor', type=str, default='val_loss', help='value to monitor')
    parser.add_argument('--random_seed', type=int, help='random seed to parameterize')
    parser.add_argument('--patience', type=int, default=6, help='early stoping parameter')
    parser.add_argument('--verbose', action='store_true', help='whether to add extra logging')
    parser.add_argument('--save_checkpoint', action='store_true', help='whether to save checkpoints')
    parser.add_argument('--log_generative_metrics', action='store_true', help='whether evaluate FID, KID')
    parser.add_argument('--enable_kid', action='store_true', help='whether to evaluate KID')
    parser.add_argument('--disable_early_stop', action='store_true', help='whether disable early stopping behavior')

    return parser
