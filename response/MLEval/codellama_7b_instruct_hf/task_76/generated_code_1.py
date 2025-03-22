import tensorflow as tf

def method():
    # Create a simple graph with two operations
    graph = tf.Graph()
    with graph.as_default():
        op1 = tf.constant(1, name="op1")
        op2 = tf.constant(2, name="op2")

    # Use the get_name() method to get the name of each operation
    op1_name = op1.get_name()
    op2_name = op2.get_name()

    # Print the names of the operations
    print(f"Operation 1 name: {op1_name}")
    print(f"Operation 2 name: {op2_name}")

    # Return the final output
    return None

# Call the method for validation
output = method()