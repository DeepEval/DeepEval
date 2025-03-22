import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense

def dl_model():
    # Input layer for CIFAR-10 images (32x32 pixels, 3 channels)
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    # Concatenate input and output of the first layer
    concat1 = Concatenate(axis=-1)([input_layer, conv1])
    
    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat1)
    # Concatenate input and output of the second layer
    concat2 = Concatenate(axis=-1)([concat1, conv2])
    
    # Third convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat2)
    
    # Flatten the final output for the fully connected layers
    flatten_layer = Flatten()(conv3)
    
    # First fully connected layer
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    # Second fully connected layer
    dense2 = Dense(units=128, activation='relu')(dense1)
    # Output layer with softmax activation for multi-class classification
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Constructing the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Usage
model = dl_model()
model.summary()