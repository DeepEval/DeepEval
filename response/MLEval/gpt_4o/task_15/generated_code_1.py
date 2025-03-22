import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def method():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize the input data
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Convert labels to categorical one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # Create a simple neural network model
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Experiment with different batch sizes and epochs
#     batch_size_options = [32, 64, 128]
#     epoch_options = [5, 10, 15]
    
#     # Store results for comparison
#     results = {}
    
#     for batch_size in batch_size_options:
#         for epochs in epoch_options:
#             print(f"Training with batch_size={batch_size}, epochs={epochs}")
#             history = model.fit(x_train, y_train,
#                                 batch_size=batch_size,
#                                 epochs=epochs,
#                                 validation_data=(x_test, y_test),
#                                 verbose=0)  # Set verbose to 0 for concise output
            
#             # Evaluate the model
#             test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
#             results[(batch_size, epochs)] = test_accuracy
#             print(f"Test Accuracy: {test_accuracy:.4f}")
    
#     # Find the best configuration
#     best_config = max(results, key=results.get)
#     best_accuracy = results[best_config]
    
#     print(f"Best configuration: batch_size={best_config[0]}, epochs={best_config[1]} with accuracy={best_accuracy:.4f}")
    
#     output = {
#         "best_config": best_config,
#         "best_accuracy": best_accuracy,
#         "results": results
#     }
    
#     return output

# # Call the method for validation
# output = method()
method()