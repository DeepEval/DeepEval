import tensorflow as tf
from nncf import NNCFConfig
import nncf.torch  # Assuming this is the correct import for nncf's create_compressed_model function
from nncf.torch.nncf_config import CompressionState  # Import the CompressionState class

def create_compressed_model_and_algo_for_test(model, config, compression_state=None, force_no_init=False):
    assert isinstance(config, NNCFConfig), "The config input must be an instance of NNCFConfig"
    tf.keras.backend.clear_session()
    if force_no_init:
        compression_state = {}
    compressed_model, algo = nncf.torch.create_compressed_model(model, config, compression_state)
    return compressed_model, algo

if __name__ == "__main__":
    # Create a sample model
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(10), tf.keras.layers.Dense(10)])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Create a sample NNCFConfig object
    config = NNCFConfig()

    # Create a sample compression state object
    compression_state = CompressionState()

    # Create sample inputs
    force_no_init = True

    # Call the function and print the results
    compressed_model, algo = create_compressed_model_and_algo_for_test(model, config, compression_state, force_no_init)
    print("Compressed Model:", compressed_model)
    print("Algorithm:", algo)