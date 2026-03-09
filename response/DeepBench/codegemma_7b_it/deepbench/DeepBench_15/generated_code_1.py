from typing import Optional, Tuple

class NNCFConfig:
    # ...

    def _create_input_info(sample_sizes: Tuple[int, int, int, int]) -> list[dict]:
        return [{"sample_size": ss} for ss in sample_sizes]

    def get_empty_config(
        model_size: int = 4,
        input_sample_sizes: Optional[Tuple[int, int, int, int]] = None,
        input_info: Optional[dict] = None,
    ) -> NNCFConfig:
        """Generates a NNCFConfig instance with empty model and input info."""
        if input_sample_sizes is None:
            input_sample_sizes = [1, 1, 4, 4]
        if input_info is None:
            input_info = _create_input_info(input_sample_sizes)
        return NNCFConfig(
            model="empty_config", model_size=model_size, input_info=input_info
        )

if __name__ == "__main__":
    # Example test case: create inputs, call the function, and print results/assert
    config = get_empty_config()
    print(config.model)  # Output: empty_config
    print(config.model_size)  # Output: 4
    print(config.input_info)  # Output: [{''sample_size'': 1}, {''sample_size'':
  1}, {''sample_size'': 4}, {''sample_size'': 4}]