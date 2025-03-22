import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

    # Block 2
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(max_pooling1)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Combine paths
    combined = Concatenate()([input_layer, max_pooling1, max_pooling2])

    # Fully connected layer
    flatten_layer = Flatten()(combined)
    dense = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense)

    return model