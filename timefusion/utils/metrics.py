
from torch import Tensor
import numpy as np

from typing import Union, Generator

def windowed_product(stop: int, window: int) -> Generator:
    for x1 in range(stop):
        for x2 in range(x1, min(x1 + window, stop)):
            yield x1, x2

def variogram_score(
    predictions: Tensor, 
    realisations: Tensor, 
    p: Union[float,int] = 0.5,
    **kwargs
) -> float:

    """
    Args:
        predictions: [num samples, num timeseries, num timesteps]
        realisations: [num timeseries, num timesteps]
    Returns:
        float: Variogram score
    Reference:
        https://journals.ametsoc.org/view/journals/mwre/143/4/mwr-d-14-00269.1.xml
    """

    timeseries, timesteps = realisations.shape

    match kwargs.get("weights",None):
        case "local":
            window_size = kwargs.get("window_size",1)
            variogram_sum = 0
            sum_elements = 0
            for i, j in windowed_product(timesteps, window_size):
                repeated_pi = predictions[...,i].unsqueeze(-1).expand(-1,-1,timeseries)
                repeated_pj = predictions[...,j].unsqueeze(-1).expand(-1,-1,timeseries)
                repeated_ri = realisations[...,i].unsqueeze(-1).expand(-1,timeseries)
                repeated_rj = realisations[...,j].unsqueeze(-1).expand(-1,timeseries)

                variogram_sum += float(((abs(repeated_ri - repeated_rj.T)**p - (abs(repeated_pi - repeated_pj.transpose(2,1))**p).mean(dim=0))**2).mean())
                sum_elements += 1 

            return variogram_sum / sum_elements
        case _:
            flat_p = predictions.flatten(start_dim=1)
            flat_r = realisations.flatten()

            repeated_p = predictions.unsqueeze(-1).expand(-1,-1,flat_p.shape[1])
            repeated_r = realisations.unsqueeze(-1).expand(-1,flat_r.shape[0])

            return ((abs(repeated_r - repeated_r.T)**p - (abs(repeated_p - repeated_p.transpose(2,1))**p).mean(dim=0))**2).mean()


def crps(x: Tensor, y: float) -> float:
    """
    Args:
        x: [num samples], predicted values
        y: realized value
    Returns:
        crps: Continuous Ranked Probability Score
    """
    repeated_x = x.unsqueeze(-1).expand(-1,x.shape[0])

    # Calculate crps using NRG expression
    return float(abs(x - y).mean() - 1/2 * abs(repeated_x - repeated_x.T).mean())


def crps_sum(predictions: Tensor, realisations: Tensor) -> float:
    """
    Args:
        predictions: [num samples, num timeseries, num timesteps]
        realisations: [num timeseries, num timesteps]
    """

    # Sum predictions and realisation over all timeseries
    x = predictions.sum(dim=1)
    y = realisations.sum(dim=0)

    return np.mean([crps(x[:,i],y[i]) for i in range(len(y))])
