import tensorflow as tf
from nncf import NNCFConfig
from nncf.tensorflow import create_compressed_model

def create_compressed_model_and_algo_for_test(model, config, compression_state=None, force_no_init=False):
    assert isinstance(config, NNCFConfig), "config must be an instance of NNCFConfig"
    
    tf.keras.backend.clear_session()

    if force_no_init:
        compression_state = {}

    compressed_model, compression_algorithm = create_compressed_model(model, config, compression_state)
    
    return compressed_model, compression_algorithm

if __name__ == "__main__":
    # Create a dummy TensorFlow model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=(20,), activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    # Create a dummy NNCFConfig object (using a dictionary for illustration purposes)
    config_dict = {
        "input_info": {
            "sample_size": [1, 20]
        },
        "compression": [
            {
                "algorithm": "quantization"
            }
        ]
    }
    config = NNCFConfig(config_dict)

    # Call the function with example inputs
    compressed_model, compression_algorithm = create_compressed_model_and_algo_for_test(model, config, force_no_init=True)

    # Print the results
    print(compressed_model)
    print(compression_algorithm)