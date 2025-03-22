from keras.layers import Input, Conv2D, Add, Flatten, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical
import keras.backend as K

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)  # Increase feature map width
    x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)  # Restore number of channels
    
    # Branch path (directly connecting to the input)
    branch = input_layer
    
    # Combine the main path and the branch path with addition
    combined = Add()([x, branch])
    
    # Flatten the output and add a fully connected layer
    flat = Flatten()(combined)
    output_layer = Dense(10, activation='softmax')(flat)  # 10 classes for MNIST
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage
if __name__ == "__main__":
    # Load the MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Preprocess the data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train[..., K.image_data_format() == 'channels_first' and 0 or 1]
    x_test = x_test[..., K.image_data_format() == 'channels_first' and 0 or 1]

    # One-hot encode the labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # Build the model
    model = dl_model()
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Print the model summary
    model.summary()
    
    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))