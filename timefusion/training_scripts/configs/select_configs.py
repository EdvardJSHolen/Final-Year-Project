import random
import json
import numpy as np
from typing import Dict, Any

num_trials = 50
<<<<<<< HEAD:timefusion/training_scripts/configs/select_configs.py
random_seed = 15
config_name = f"electricity_{random_seed}"
=======
random_seed = 0
config_name = f"timefusion_{random_seed}"
>>>>>>> 1d82e864ac9e61ecaf97bf53fb3b33544f57e898:TimeFusion/training_scripts/configs/select_configs.py

def get_hyperparameters(ranges: Dict[str,Any], parameter_id: int) -> Dict[str,Any]:
    """
    Returns a dictionary of hyperparameters for a given parameter_id.
    """

    div_terms = list(np.cumprod([len(value) for _, value in ranges]))
    div_terms.insert(0,1)
    div_terms.pop(-1)

    return {ranges[i][0]:ranges[i][1][int(parameter_id // div) % len(ranges[i][1])] for i, div in enumerate(div_terms)}


# Get ranges from json file
ranges = json.load(open(f"ranges.json","r"))

# Randomly select combinations to test
random.seed(random_seed)
max_parameter_id = np.prod([len(value) for _, value in ranges])
parameters_ids = random.sample(range(int(max_parameter_id)),num_trials)

# Get the combinations corresponding to the selected parameter ids
configurations = [get_hyperparameters(ranges,parameter_id) for parameter_id in parameters_ids]

# Save the configurations
json.dump(configurations,open(f"{config_name}.json","w"),indent=4)
