from nncf import NNCFConfig
from nncf.compression import create_compressed_model

def create_compressed_model_and_algo_for_test(model, config, compression_state=None, force_no_init=False):
  assert isinstance(config, NNCFConfig)
  import tensorflow as tf
  tf.keras.backend.clear_session()
  if force_no_init:
    compression_state = {}
  return create_compressed_model(model, config, compression_state)

if __name__ == "__main__":
  model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)),
      tf.keras.layers.Dense(1, activation='sigmoid')
  ])
  config = NNCFConfig(compression_algorithm="none")
  compressed_model, algo = create_compressed_model_and_algo_for_test(model, config)
  print("Compressed Model:", compressed_model)
  print("Algorithm:", algo)