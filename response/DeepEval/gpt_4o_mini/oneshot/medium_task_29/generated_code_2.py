import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have a shape of 32x32 with 3 channels (RGB)

    # Define the max pooling layers with specified window sizes and strides
    max_pooling1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    max_pooling3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)

    # Flatten the outputs of each max pooling layer
    flatten1 = Flatten()(max_pooling1)
    flatten2 = Flatten()(max_pooling2)
    flatten3 = Flatten()(max_pooling3)

    # Concatenate the flattened outputs
    concatenated = Concatenate()([flatten1, flatten2, flatten3])

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(concatenated)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Output layer with 10 classes for CIFAR-10
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model