from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Flatten, Dense, Input
from tensorflow.keras.initializers import HeNormal

def dl_model():
    inputs = Input(shape=(28, 28, 1))
    x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding="same", use_bias=False, kernel_initializer=HeNormal())(inputs)
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=False, kernel_initializer=HeNormal())(x)
    x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding="same", use_bias=False, kernel_initializer=HeNormal())(x)
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=False, kernel_initializer=HeNormal())(x)
    x = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding="same", use_bias=False, kernel_initializer=HeNormal())(x)
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=False, kernel_initializer=HeNormal())(x)
    x = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding="same", use_bias=False, kernel_initializer=HeNormal())(x)

    outputs = Flatten()(x)
    outputs = Dense(units=10, activation="softmax")(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    return model