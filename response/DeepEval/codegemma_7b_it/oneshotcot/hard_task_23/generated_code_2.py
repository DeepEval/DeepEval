import keras
from keras.models import Model
from keras.layers import (
    Input,
    Conv2D,
    AvgPooling2D,
    Conv2DTranspose,
    concatenate,
    BatchNormalization,
    Flatten,
    Dense,
)

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv_init = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Local feature extraction branch
    branch_local = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_init)
    branch_local = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_local)

    # Downsampling branch
    branch_down = AvgPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv_init)
    branch_down = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_down)

    # Upsampling branch
    branch_up = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv_init)
    branch_up = Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(branch_up)

    # Concatenation and refinement
    concat = concatenate([branch_local, branch_down, branch_up])
    concat = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)

    # Fully connected layer
    flatten_layer = Flatten()(concat)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Model creation
    model = Model(inputs=input_layer, outputs=dense_layer)

    return model