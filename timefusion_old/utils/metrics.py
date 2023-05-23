
import numpy as np
import scipy.integrate as integrate

from itertools import combinations, product
from scipy.stats import percentileofscore
from typing import Union, Generator

def windowed_product(stop: int, window: int) -> Generator:
    for x1 in range(stop):
        for x2 in range(x1, min(x1 + window, stop)):
            yield x1, x2

def variogram_score(
        realisations: np.ndarray, 
        predictions: np.ndarray, 
        p: Union[float,int] = 0.5,
        **kwargs
    ) -> float:

    """
        https://journals.ametsoc.org/view/journals/mwre/143/4/mwr-d-14-00269.1.xml
        realisations: [num timeseries, num timesteps] or [num timesteps]
        predictions: [num samples, num timeseries, num timesteps] or [num samples, num timestamps]
    """
    
    # Ensure that the passed realisations and predictions are of correct structure
    assert (realisations.ndim,predictions.ndim) in {(1,2),(2,3)}
    assert  realisations.shape == predictions.shape[1:]

    if realisations.ndim == 1:
        realisations = np.expand_dims(realisations,0)
        predictions = np.expand_dims(predictions,1)

    num_timeseries = realisations.shape[0]
    num_timesteps = realisations.shape[1]

    match kwargs.get("weights",None):
        case "local":
            window_size = kwargs.get("window_size",1)
            variogram_sum = 0
            for x1, x2 in windowed_product(num_timesteps, window_size):
                weight =  1 - (abs(x2 - x1)/window_size)**2
                for y1, y2 in combinations(range(num_timeseries),2):
                    variogram_sum += weight * ((abs(realisations[y1,x1] - realisations[y2,x2])**p - np.mean(abs(predictions[:,y1,x1] - predictions[:,y2,x2])**p))**2)
        case "inverse":
            variogram_sum = 0
            for x1, x2 in product(range(num_timesteps), repeat=2):
                weight = 1 / (1 + abs(x1 - x2)) # Slight generalisation of original paper used here
                for y1, y2 in combinations(range(num_timeseries),2):
                    variogram_sum += weight * ((abs(realisations[y1,x1] - realisations[y2,x2])**p - np.mean(abs(predictions[:,y1,x1] - predictions[:,y2,x2])**p))**2)
        case _:
            variogram_sum = 0
            for x1, x2 in product(range(num_timesteps), repeat=2):
                for y1, y2 in combinations(range(num_timeseries),2):
                    variogram_sum += (abs(realisations[y1,x1] - realisations[y2,x2])**p - np.mean(abs(predictions[:,y1,x1] - predictions[:,y2,x2])**p))**2

    return variogram_sum

def crps(F: np.ndarray, x: float) -> float:
    """
        F: [num samples]
    """
    return integrate.quad(lambda y: (percentileofscore(F,y,kind="weak")/100 - np.heaviside(y - x, 1))**2, min(min(F),x), max(max(F),x),epsrel=1.49e-2,limit=100)


def crps_sum(realisations: np.ndarray, predictions: np.ndarray) -> float:
    """
        https://arxiv.org/pdf/1910.03002.pdf
        realisations: [num timeseries, num timesteps]
        predictions: [num samples, num timeseries, num timesteps]
    """
    F = np.sum(predictions, axis=1)
    x = np.sum(realisations, axis = 0)

    return np.mean([crps(F[:,i],x[i]) for i in range(len(x))])

