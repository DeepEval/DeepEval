from nncf.config import NNCFConfig

def get_empty_config(model_size=4, input_sample_sizes=None, input_info=None):
    if input_sample_sizes is None:
        input_sample_sizes = [1, 1, 4, 4]

    def _create_input_info(input_sample_sizes):
        return [{"sample_size": size} for size in input_sample_sizes]

    nncf_config = NNCFConfig()
    nncf_config.update({
        "model": "empty_config",
        "model_size": model_size
    })
    
    if input_info:
        nncf_config.update({"input_info": input_info})
    else:
        nncf_config.update({"input_info": _create_input_info(input_sample_sizes)})
    
    return nncf_config

if __name__ == "__main__":
    model_size = 8
    input_sample_sizes = [2, 2, 8, 8]
    input_info = [{"sample_size": size} for size in input_sample_sizes]
    
    config = get_empty_config(model_size, input_sample_sizes, input_info)
    print(config)
    
    config = get_empty_config(model_size, input_sample_sizes)
    print(config)
    
    config = get_empty_config(model_size)
    print(config)
    
    config = get_empty_config()
    print(config)