import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Third convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    
    # Separate convolutional layer for input
    conv4 = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    
    # Concatenate the outputs of all convolutional layers
    concat_layer = Concatenate(axis=-1)([conv1, conv2, conv3, conv4])
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(concat_layer)
    flatten = Flatten()(batch_norm)
    
    # First fully connected layer
    dense1 = Dense(units=256, activation='relu')(flatten)
    
    # Second fully connected layer
    dense2 = Dense(units=128, activation='relu')(dense1)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model