import tensorflow as tf

def method():
  # Create some tensors
  tensor1 = tf.constant([1, 2, 3])
  tensor2 = tf.constant([4, 5, 6])

  # Addition
  addition = tensor1 + tensor2
  
  # Subtraction
  subtraction = tensor1 - tensor2

  # Multiplication
  multiplication = tensor1 * tensor2

  # Division
  division = tensor1 / tensor2

  # Dot product
  dot_product = tf.reduce_sum(tensor1 * tensor2)

  # Concatenation
  concatenation = tf.concat([tensor1, tensor2], axis=0)

  # Output
  output = {"addition": addition,
            "subtraction": subtraction,
            "multiplication": multiplication,
            "division": division,
            "dot_product": dot_product,
            "concatenation": concatenation}
  return output

# Call the method and print the output
result = method()
print(result)