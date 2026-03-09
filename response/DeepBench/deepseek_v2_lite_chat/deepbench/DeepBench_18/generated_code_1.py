import tensorflow as tf
from nncf import NNCFConfig

def create_compressed_model_and_algo_for_test(model, config, compression_state=None, force_no_init=False):
    assert isinstance(config, NNCFConfig), "The config should be an instance of NNCFConfig"
    
    # Clear the TensorFlow backend session
    tf.keras.backend.clear_session()
    
    if force_no_init:
        compression_state = {}
    
    # Create a compressed model and algorithm
    compressed_model, compression_algorithm = create_compressed_model(model, config, compression_state)
    
    return compressed_model, compression_algorithm

def run_example():
    # Create sample input values
    input_tensor = tf.random.normal([1, 10, 10, 3])
    
    # Call the function
    compressed_model, compression_algorithm = create_compressed_model_and_algo_for_test(model, config, compression_state=None, force_no_init=True)
    
    # Dummy output for the example
    print("Compressed model:", compressed_model)
    print("Compression algorithm:", compression_algorithm)

if __name__ == "__main__":
    # Run the example
    run_example()