import nncf

def get_empty_config(model_size=4, input_sample_sizes=None, input_info=None):
    if input_sample_sizes is None:
        input_sample_sizes = [1, 1, 4, 4]

    def _create_input_info(input_sample_sizes):
        input_info = []
        for sample_size in input_sample_sizes:
            input_info.append({"sample_size": sample_size})
        return input_info

    config = nncf.NNCFConfig()
    config.model = "empty_config"
    config.model_size = model_size
    if input_info is not None:
        config.input_info = input_info
    else:
        config.input_info = _create_input_info(input_sample_sizes)
    return config

if __name__ == "__main__":
    # Run the example
    config = get_empty_config(model_size=8, input_sample_sizes=[1, 2, 3, 4], input_info={"sample_size": 4})
    print(config)