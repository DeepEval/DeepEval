import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Add, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # First block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

    # Second block
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pooling1)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Combine paths
    concat = Concatenate()([max_pooling1, max_pooling2])

    # Additional convolution layer
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(concat)

    # Output layer
    flatten_layer = Flatten()(conv3)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model