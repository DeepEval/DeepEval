from nncf import NNCFConfig

def get_empty_config(model_size=4, input_sample_sizes=None, input_info=None):
    def _create_input_info(sample_sizes):
        return [{"sample_size": size} for size in sample_sizes]

    if input_sample_sizes is None:
        input_sample_sizes = [1, 1, 4, 4]

    input_info_instance = input_info if input_info is not None else _create_input_info(input_sample_sizes)

    config = NNCFConfig()
    config.update({"model": "empty_config", "model_size": model_size, "input_info": input_info_instance})

    return config

if __name__ == "__main__":
    config = get_empty_config()
    print(config)