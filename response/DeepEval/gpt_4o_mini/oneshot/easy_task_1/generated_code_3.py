import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Input layer for 28x28 grayscale images
    input_layer = Input(shape=(28, 28, 1))
    
    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    
    # First max pooling layer
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    
    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pool1)

    # Flatten the feature maps to a 1D vector
    flatten_layer = Flatten()(conv2)

    # First fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Second fully connected layer
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer with softmax activation for classification into 10 classes
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of how to instantiate the model
model = dl_model()
model.summary()