import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have shape 32x32x3
    
    # Convolutional paths
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pool3x3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # Concatenating the outputs of the convolutional layers and max pooling
    concatenated = Concatenate()([conv1x1, conv3x3, conv5x5, max_pool3x3])

    # Flattening the concatenated output
    flatten_layer = Flatten()(concatenated)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)  # 10 classes for CIFAR-10

    # Constructing the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model