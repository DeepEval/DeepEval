import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from nncf.api.compression import CompressionLevel
from nncf.tensorflow.algorithm_selector import TF_COMPRESSION_ALGORITHMS

def create_compressed_model_and_algo_for_test(model, config, compression_state, force_no_init):
    assert isinstance(config, NNCFConfig)

    # Clear the TensorFlow backend session
    tf.keras.backend.clear_session()

    if force_no_init:
        # Set the compression_state to an empty dictionary
        compression_state = {}

    # Create a compressed model and algorithm
    compressed_model, compression_algorithm = create_compressed_model(model, config, compression_state)

    return compressed_model, compression_algorithm

# Example usage:
if __name__ == "__main__":
    # Create a sample model
    model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))

    # Create a sample NNCFConfig object
    config = NNCFConfig()
    config.from_json('./config.json')

    # Create a sample compression_state object
    compression_state = {
        'weights': {
            'magnitude': 0.5,
            'sparsity': 0.1
        },
        'activations': {
            'magnitude': 0.5,
            'sparsity': 0.1
        }
    }

    # Call the function
    compressed_model, compression_algorithm = create_compressed_model_and_algo_for_test(model, config, compression_state, force_no_init=True)

    # Print the results
    print(compressed_model.summary())
    print(compression_algorithm.get_configuration())