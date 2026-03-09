import tensorflow as tf
from nncf import NNCFConfig, create_compressed_model

def create_compressed_model_and_algo_for_test(model, config, compression_state=None, force_no_init=False):
    assert isinstance(config, NNCFConfig), "config must be an instance of NNCFConfig"
    
    tf.keras.backend.clear_session()
    
    if force_no_init:
        compression_state = {}
    
    compressed_model = create_compressed_model(model, config, compression_state)
    
    return compressed_model, None  # The algorithm is not returned in this function

if __name__ == "__main__":
    # Example usage
    # Assuming we have a simple Keras model and a sample NNCFConfig
    
    # Sample Keras model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=(5,), activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Sample NNCFConfig
    config = NNCFConfig()
    config.update({"compression": {"algorithm": "quantization"}})
    
    # Call the function
    compressed_model, _ = create_compressed_model_and_algo_for_test(model, config)
    
    # Print the results (model summary)
    compressed_model.summary()