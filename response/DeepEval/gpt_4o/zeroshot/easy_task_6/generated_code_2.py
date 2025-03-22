from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    # First convolutional layer: increases the feature map width
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    # Second convolutional layer: restores the number of channels
    x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
    
    # Branch path directly connected to the input
    branch = input_layer
    
    # Combine the two paths through an addition operation
    combined = Add()([x, branch])
    
    # Flatten and fully connected layer for final classification
    flat = Flatten()(combined)
    output = Dense(10, activation='softmax')(flat)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Example usage
if __name__ == "__main__":
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # Build the model
    model = dl_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.2)
    
    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test loss: {loss}, Test accuracy: {accuracy}")