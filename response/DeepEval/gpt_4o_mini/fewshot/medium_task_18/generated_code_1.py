import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Different convolutional paths
    conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    conv_3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv_5x5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(input_layer)
    
    # Max pooling path
    max_pooling = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # Concatenating the outputs
    concatenated = Concatenate()([conv_1x1, conv_3x3, conv_5x5, max_pooling])

    # Flattening the concatenated output
    flatten_layer = Flatten()(concatenated)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    # Creating the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model