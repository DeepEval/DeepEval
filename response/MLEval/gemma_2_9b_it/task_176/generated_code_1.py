import tensorflow as tf

def method():
  logits = tf.constant([1.0, 2.0, 3.0]) 
  output = tf.nn.softmax(logits)
  return output

output = method()
print(output)