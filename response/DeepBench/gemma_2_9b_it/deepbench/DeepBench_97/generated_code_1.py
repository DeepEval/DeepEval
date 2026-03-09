import numpy as np

def softmax(x, temperature, axis=-1, shift=False):
  if shift:
    x = x - np.max(x, axis=axis, keepdims=True)
  return np.exp(x / temperature) / np.sum(np.exp(x / temperature), axis=axis, keepdims=True)

if __name__ == "__main__":
  x = np.array([1.0, 2.0, 3.0])
  temperature = 1.0
  result = softmax(x, temperature)
  print(result)