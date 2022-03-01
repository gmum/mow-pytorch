from typing import Callable
from metrics.rec_err import mean_per_image_se
import torch
from metrics.cw import cw_normality
from common.noise_creator import NoiseCreator
from metrics.mmd import mmd_wrapper


def get_cost_function(model: str, noise_creator: NoiseCreator) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    cost_functions: dict[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]] = {
        'ae': lambda x, _, y: (mean_per_image_se(x, y), torch.FloatTensor([0.0]).to(x.device)),
        'cwae': lambda x, z, y: (mean_per_image_se(x, y), torch.log(cw_normality(z))),
        'wae-mmd': lambda x, z, y: (mean_per_image_se(x, y), mmd_wrapper(z, noise_creator)),
    }

    return cost_functions[model]
