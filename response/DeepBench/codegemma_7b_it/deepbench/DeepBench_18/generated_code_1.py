from neural_tangents import NNCFConfig
from neural_tangents.algorithms import create_compressed_model
from neural_tangents.utils.misc import clear_tf_backend_session

def create_compressed_model_and_algo_for_test(model, config, compression_state=None, force_no_init=False):
    assert isinstance(config, NNCFConfig), "config must be an instance of NNCFConfig"
    clear_tf_backend_session()
    if force_no_init:
        compression_state = {}
    compressed_model, algo = create_compressed_model(model, config, compression_state=compression_state)
    return compressed_model, algo

if __name__ == "__main__":
    # Example usage
    import tensorflow as tf

    # Create a model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation="relu", input_shape=(784,)),
        tf.keras.layers.Dense(10)
    ])

    # Create a NNCFConfig object
    config = NNCFConfig(
        model_size_per_layer=100,
        num_components=5,
        num_samples=10000,
        lr=0.01,
        num_steps=100000,
    )

    # Call the function
    compressed_model, algo = create_compressed_model_and_algo_for_test(model, config)

    # Print the results
    print(compressed_model)
    print(algo)