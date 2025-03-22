from keras.models import Model
from keras.layers import Input, Conv2D, Concatenate, Add, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Feature extraction path 1: 1x1 convolution
    conv1_1x1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(inputs)

    # Feature extraction path 2: 1x1, 1x7, 7x1 convolutions
    conv1_1x1_2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(inputs)
    conv1_1x7_2 = Conv2D(filters=64, kernel_size=(1, 7), activation='relu')(conv1_1x1_2)
    conv1_7x1_2 = Conv2D(filters=64, kernel_size=(7, 1), activation='relu')(conv1_1x7_2)

    # Concatenate outputs from both paths
    concat = Concatenate()([conv1_1x1, conv1_7x1_2])

    # Main path: 1x1 convolution and addition with branch
    conv1_1x1_3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(concat)
    branch = Input(shape=(32, 32, 3))
    added = Add()([conv1_1x1_3, branch])

    # Classification layers
    flatten = Flatten()(added)
    dense = Dense(units=64, activation='relu')(flatten)
    outputs = Dense(units=10, activation='softmax')(dense)

    # Model creation
    model = Model(inputs=[inputs, branch], outputs=outputs)

    return model