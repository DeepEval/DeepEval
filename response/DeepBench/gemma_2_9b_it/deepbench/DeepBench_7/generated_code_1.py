import numpy as np
from neural_compressor.api import NNCFConfig
from neural_compressor.optim import CompressionAlgorithmController

def create_compressed_model_and_algo_for_test(model, nncf_config=None, dummy_forward=None, wrap_inputs=None, compression_state=None):
  if nncf_config is not None:
    nncf_config.validate()
  compressed_model = model
  algo_controller = CompressionAlgorithmController(
      model=compressed_model,
      config=nncf_config,
      dummy_forward=dummy_forward,
      wrap_inputs=wrap_inputs,
      compression_state=compression_state
  )
  return compressed_model, algo_controller

if __name__ == "__main__":
  model = np.random.rand(10, 10)
  nncf_config = NNCFConfig(
      compression="weight_quantization",
      quantization_config=dict(
          type="dynamic",
          bits=4
      )
  )
  dummy_forward = lambda x: x
  wrap_inputs = lambda x: x
  compression_state = {}
  compressed_model, algo_controller = create_compressed_model_and_algo_for_test(
      model, nncf_config, dummy_forward, wrap_inputs, compression_state
  )
  print("Compressed Model Shape:", compressed_model.shape)
  print("Algorithm Controller:", algo_controller)