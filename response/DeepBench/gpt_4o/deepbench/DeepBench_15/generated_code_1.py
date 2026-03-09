from typing import List, Dict, Optional, Union

class NNCFConfig:
    def __init__(self):
        self.config = {}

    def update(self, new_config: Dict):
        self.config.update(new_config)

    def __repr__(self):
        return repr(self.config)

def get_empty_config(model_size: int = 4, 
                     input_sample_sizes: Optional[Union[List[int], tuple]] = None, 
                     input_info: Optional[Dict] = None) -> NNCFConfig:
    
    if input_sample_sizes is None:
        input_sample_sizes = [1, 1, 4, 4]

    def _create_input_info() -> List[Dict[str, Union[List[int], tuple]]]:
        return [{"sample_size": input_sample_sizes}]

    nncf_config = NNCFConfig()
    nncf_config.update({
        "model": "empty_config",
        "model_size": model_size,
        "input_info": input_info if input_info is not None else _create_input_info()
    })

    return nncf_config

if __name__ == "__main__":
    # Sample input values
    config1 = get_empty_config()
    print(config1)

    config2 = get_empty_config(model_size=8, input_sample_sizes=[2, 2, 8, 8])
    print(config2)

    config3 = get_empty_config(input_info={"custom_key": "custom_value"})
    print(config3)