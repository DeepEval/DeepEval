from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Max pooling layers with different scales
    x = input_layer
    for i in range(3):
        x = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
    x = Concatenate()(x)

    # Block 2: Separable convolutional layers with varying kernel sizes
    x = Lambda(lambda x: tf.split(x, 4, axis=-1))(x)
    for i in range(4):
        x = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(x[i])
        x = BatchNormalization()(x)
        x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
    x = Concatenate()(x)

    # Output layer
    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=10, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)

    return model