import tensorflow as tf

def method():
    # Create a simple computational graph
    # Note: This example uses TensorFlow 1.x style, which is better suited for graph manipulation.
    # In TensorFlow 2.x, eager execution is the default, and some adjustments might be needed.
    
    # Create a new graph
    graph = tf.Graph()
    
    with graph.as_default():
        # Define some operations
        a = tf.constant(2, name='const_a')
        b = tf.constant(3, name='const_b')
        c = tf.add(a, b, name='add_c')
    
    # To inspect the operations, use the graph's operations
    output = []
    for op in graph.get_operations():
        # Accessing the name of each operation
        output.append(op.name)
        print(f"Operation name: {op.name}")

    return output

# Call the method for validation
operations_names = method()
print("Operations in the graph:", operations_names)