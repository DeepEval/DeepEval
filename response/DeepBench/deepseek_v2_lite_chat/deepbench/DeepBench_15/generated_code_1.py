import nncf

def get_empty_config(model_size=4, input_sample_sizes=None, input_info=None):
    def _create_input_info(sizes):
        return [{'sample_size': size} for size in sizes]

    input_info = _create_input_info(input_sample_sizes if input_sample_sizes else [1, 1, 4, 4])

    config = nncf.Config(model="empty_config", model_size=model_size, input_info=input_info)

    return config

if __name__ == "__main__":
    model_size = 4
    input_sizes = [1, 2, 4, 8]
    input_info = get_empty_config(model_size, input_sizes)
    print(input_info)