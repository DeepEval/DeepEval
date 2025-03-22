from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    input_shape = (28, 28, 1)  # MNIST images have a shape of 28x28 with a single channel

    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Function to create the block
    def create_branch_block(input_tensor):
        # 1x1 Convolutional layer to elevate dimensions
        x = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        
        # 3x3 Depthwise Separable Convolutional layer
        x = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(x)
        
        # 1x1 Convolutional layer to reduce dimensions
        x = Conv2D(filters=input_tensor.shape[-1], kernel_size=(1, 1), activation='relu')(x)
        
        # Adding the block's input to the output (skip connection)
        x = Add()([x, input_tensor])
        
        return x

    # Create three branches with the same block structure
    branch1 = create_branch_block(input_layer)
    branch2 = create_branch_block(input_layer)
    branch3 = create_branch_block(input_layer)

    # Concatenate outputs from the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Flattening layer
    flattened = Flatten()(concatenated)

    # Fully connected layer to produce classification probabilities for 10 classes
    output_layer = Dense(units=10, activation='softmax')(flattened)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage of the model
if __name__ == "__main__":
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Preprocess data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train[..., None]  # add channel dimension
    x_test = x_test[..., None]  # add channel dimension
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Get the model
    model = dl_model()

    # Print model summary
    model.summary()

    # Train the model (for demonstration, we use a small number of epochs)
    model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.2)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_accuracy}")