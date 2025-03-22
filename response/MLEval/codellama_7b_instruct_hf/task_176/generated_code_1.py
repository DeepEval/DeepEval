import tensorflow as tf

def method(logits):
    softmax_activations = tf.nn.softmax(logits)
    return softmax_activations

# Test the method
logits = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
output = method(logits)
print(output)


tf.Tensor([[0.09003057 0.24472847 0.6651443 ],
        [0.01885232 0.03651459 0.9364851 ]], shape=(2, 3), dtype=float32)