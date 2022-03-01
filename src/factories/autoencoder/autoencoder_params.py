from common.args_parser import BaseArguments


class AutoEncoderParams(BaseArguments):
    latent_dim: int
    lambda_val: float
    load_checkpoint: str
