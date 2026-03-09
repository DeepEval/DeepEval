import numpy as np
from nncf import NNCFConfig

def get_empty_config(model_size=4, input_sample_sizes=None, input_info=None):
    if input_sample_sizes is None:
        input_sample_sizes = [1, 1, 4, 4]

    def _create_input_info():
        return [{"sample_size": size} for size in input_sample_sizes]

    config = NNCFConfig()
    config["model"] = "empty_config"
    config["model_size"] = model_size
    config["input_info"] = input_info if input_info is not None else _create_input_info()

    return config

if __name__ == "__main__":
    sample_input_sizes = [1, 2, 3, 4]
    config_instance = get_empty_config(model_size=8, input_sample_sizes=sample_input_sizes)
    print(config_instance)