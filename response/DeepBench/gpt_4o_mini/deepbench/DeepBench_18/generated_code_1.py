import tensorflow as tf
from nncf import NNCFConfig
from nncf.tensorflow import create_compressed_model

def create_compressed_model_and_algo_for_test(model, config, compression_state=None, force_no_init=False):
    assert isinstance(config, NNCFConfig), "config must be an instance of NNCFConfig"
    
    tf.keras.backend.clear_session()
    
    if force_no_init:
        compression_state = {}
    
    compressed_model, algo = create_compressed_model(model, config, compression_state)
    
    return compressed_model, algo

if __name__ == "__main__":
    from tensorflow.keras import layers, models
    
    # Create a sample model
    sample_model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(32,)),
        layers.Dense(10, activation='softmax')
    ])
    
    # Create a sample NNCFConfig object (this would normally be more complex and specific)
    sample_config = NNCFConfig({"compression": {"algorithm": "quantization"}})
    
    # Call the function with sample inputs
    compressed_model, algo = create_compressed_model_and_algo_for_test(sample_model, sample_config, force_no_init=True)
    
    print("Compressed Model:", compressed_model)
    print("Algorithm:", algo)