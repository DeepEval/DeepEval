from neural_compressor.config import NNCFConfig

def get_empty_config(model_size=4, input_sample_sizes=None, input_info=None):
    if input_sample_sizes is None:
        input_sample_sizes = [1, 1, 4, 4]
    
    def _create_input_info():
        return [{'sample_size': size} for size in input_sample_sizes]

    config = NNCFConfig()
    config.model = "empty_config"
    config.model_size = model_size
    config.input_info = input_info if input_info else _create_input_info()
    return config

if __name__ == "__main__":
    sample_sizes = [2, 2, 8, 8]
    custom_info = {'custom_key': 'custom_value'}
    config = get_empty_config(model_size=8, input_sample_sizes=sample_sizes, input_info=custom_info)
    print(config)