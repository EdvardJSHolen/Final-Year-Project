
import math
import copy

from torch.nn import Module

class EarlyStopper:
    def __init__(self, patience: int = 1, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.min_validation_loss = math.inf
        self.counter = 0
        self.best_weights = None

    def early_stop(self, validation_loss: float, model: Module) -> bool:

        if validation_loss > self.min_validation_loss - self.min_delta:
            self.counter += 1
        else:
            self.counter = 0

        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.best_weights = copy.deepcopy(model.state_dict())

        if self.counter >= self.patience:
            return True
        
        return False