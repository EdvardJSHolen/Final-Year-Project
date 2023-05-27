
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error
from typing import List, Dict
from torch import Tensor

from utils import metrics
from utils.data import TimeFusionDataset
from timefusion import TimeFusion

def performance(predictor: TimeFusion, data: TimeFusionDataset, indices: List, anchors: Tensor, anchor_strength: float, prediction_length: int, parameters: Dict) -> Dict:
                
    samples = predictor.sample(
        data = data,
        indices = indices,
        prediction_length = prediction_length,
        num_samples = 128,
        batch_size = 128,
        anchors = anchors,
        anchor_strength = anchor_strength,
    )
    samples = samples.cpu()

    realisations = [
        data.tensor_data[prediction_length*parameters["context_length"] + idx:prediction_length*parameters["context_length"] + idx + prediction_length, data.pred_columns].T
        for idx in indices
    ]
    realisations = torch.stack(realisations).cpu()


    mean_predictions = samples.mean(dim=1)

    return {
        "mse": mean_squared_error(realisations.flatten(), mean_predictions.flatten()),
        "mae": mean_absolute_error(realisations.flatten(), mean_predictions.flatten()),
        "mdae": median_absolute_error(realisations.flatten(), mean_predictions.flatten()),
        "crps_sum": np.mean([metrics.crps_sum(samples[i], realisations[i]) for i in range(realisations.shape[0])]),
        "variogram_score": np.mean([metrics.variogram_score(samples[i], realisations[i], weights="local", window_size=2) for i in range(realisations.shape[0])])
    }