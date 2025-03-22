import numpy as np
import tensorflow as tf

def method():
    # Create a TensorFlow tensor
    tensor = tf.constant([[1, 2, 3], [4, 5, 6]])

    # Perform basic arithmetic operations
    tensor_sum = tf.add(tensor, 5)  # Add 5 to each element
    tensor_sub = tf.subtract(tensor, 2)  # Subtract 2 from each element
    tensor_mul = tf.multiply(tensor, 3)  # Multiply each element by 3
    tensor_div = tf.divide(tensor, 2)  # Divide each element by 2

    # Perform logical operations
    tensor_equal = tf.equal(tensor, tf.constant([[1, 2, 3], [4, 5, 6]]))  # Check element-wise equality
    tensor_greater = tf.greater(tensor, tf.constant(2))  # Check element-wise greater than

    # Perform matrix operations
    tensor_transpose = tf.transpose(tensor)  # Transpose the tensor
    tensor_dot = tf.matmul(tensor, tf.constant([[7, 8], [9, 10], [11, 12]]))  # Matrix multiplication

    # Perform other operations
    tensor_abs = tf.abs(tensor)  # Absolute value
    tensor_log = tf.log(tensor)  # Natural logarithm
    tensor_exp = tf.exp(tensor)  # Exponential

    # Print the results
    print("Original Tensor:", tensor)
    print("Sum:", tensor_sum)
    print("Subtraction:", tensor_sub)
    print("Multiplication:", tensor_mul)
    print("Division:", tensor_div)
    print("Equality:", tensor_equal)
    print("Greater Than:", tensor_greater)
    print("Transposed Tensor:", tensor_transpose)
    print("Matrix Multiplication:", tensor_dot)
    print("Absolute Value:", tensor_abs)
    print("Natural Logarithm:", tensor_log)
    print("Exponential:", tensor_exp)

method()