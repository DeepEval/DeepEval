import numpy as np

def build_layer(input_size, output_size, activation='relu'):
  W = np.random.randn(input_size, output_size)
  b = np.zeros((1, output_size))
  layer = {'W': W, 'b': b, 'activation': activation}
  return layer

def forward(inputs, layer):
  z = np.dot(inputs, layer['W']) + layer['b']
  if layer['activation'] == 'relu':
    a = np.maximum(0, z)
  elif layer['activation'] == 'sigmoid':
    a = 1 / (1 + np.exp(-z))
  else:
    a = z
  return a

if __name__ == "__main__":
  input_size = 2
  output_size = 3
  layer_list = []
  layer_list.append(build_layer(input_size, output_size, activation='relu'))

  inputs = np.random.randn(1, input_size)
  for layer in layer_list:
    outputs = forward(inputs, layer)
    inputs = outputs

  print(outputs)