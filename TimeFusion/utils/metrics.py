
import numpy as np

from typing import Union, Generator

def windowed_product(stop: int, window: int) -> Generator:
    for x1 in range(stop):
        for x2 in range(x1, min(x1 + window, stop)):
            yield x1, x2

def variogram_score(
    predictions: np.ndarray, 
    realisations: np.ndarray, 
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
                repeated_pi = np.dstack([predictions[...,i]]*timeseries)
                repeated_pj = np.dstack([predictions[...,j]]*timeseries)
                repeated_ri = np.dstack([realisations[...,i]]*timeseries)[0]
                repeated_rj = np.dstack([realisations[...,j]]*timeseries)[0]

                variogram_sum += ((abs(repeated_ri - repeated_rj.T)**p - (abs(repeated_pi - repeated_pj.transpose(0,2,1))**p).mean(axis=0))**2).mean()
                sum_elements += 1 

            return variogram_sum / sum_elements
        case _:
            flat_p = predictions.reshape(predictions.shape[0],-1)
            flat_r = realisations.reshape(-1)

            repeated_p = np.dstack([flat_p]*flat_p.shape[1])
            repeated_r = np.dstack([flat_r]*flat_r.shape[0])[0]

            return ((abs(repeated_r - repeated_r.T)**p - (abs(repeated_p - repeated_p.transpose(0,2,1))**p).mean(axis=0))**2).mean()


def crps(x: np.ndarray, y: float) -> float:
    """
    Args:
        x: [num samples], predicted values
        y: realized value
    Returns:
        crps: Continuous Ranked Probability Score
    """
    repeated_x = np.dstack([x]*len(x))[0]

    # Calculate crps using NRG expression
    return abs(x - y).mean() - 1/2 * abs(repeated_x - repeated_x.T).mean()


def crps_sum(predictions: np.ndarray, realisations: np.ndarray) -> float:
    """
    Args:
        predictions: [num samples, num timeseries, num timesteps]
        realisations: [num timeseries, num timesteps]
    """

    # Sum predictions and realisation over all timeseries
    x = np.sum(predictions, axis = 1)
    y = np.sum(realisations, axis = 0)

    return np.mean([crps(x[:,i],y[i]) for i in range(len(y))])
