import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Flatten, Dense, Concatenate, BatchNormalization

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Two parallel convolutional branches with varying kernel sizes
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine features through addition
    x = Concatenate()([branch1, branch2])

    # Global average pooling layer
    x = GlobalAveragePooling2D()(x)

    # Two fully connected layers with softmax activation
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=x)

    return model