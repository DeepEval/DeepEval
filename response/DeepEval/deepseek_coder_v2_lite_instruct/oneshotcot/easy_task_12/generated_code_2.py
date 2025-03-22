import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, ReLU, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor, filters):
        x = Conv2D(filters, kernel_size=(3, 3), padding='same', depthwise_constraint=None, pointwise_constraint=None)(input_tensor)
        x = ReLU()(x)
        x = Conv2D(filters, kernel_size=(1, 1), padding='same')(x)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        return x

    # Main path
    x = block(input_tensor=input_layer, filters=32)
    y = block(input_tensor=x, filters=64)

    # Branch path
    branch = Conv2D(64, kernel_size=(1, 1), padding='same')(x)

    # Sum and flatten
    sum_layer = Add()([y, branch])
    flatten_layer = Flatten()(sum_layer)

    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model