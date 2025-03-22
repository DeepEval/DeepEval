import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dropout, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution, 3x3 convolution
    branch1_conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1_conv1)
    branch1_dropout = Dropout(rate=0.3)(branch1_conv2)

    # Branch 2: 1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution
    branch2_conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2_conv2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch2_conv1)
    branch2_conv3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch2_conv2)
    branch2_conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_conv3)
    branch2_dropout = Dropout(rate=0.3)(branch2_conv4)

    # Branch 3: Max pooling
    branch3_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    branch3_dropout = Dropout(rate=0.3)(branch3_pool)

    # Concatenate the outputs of the branches
    merged = Concatenate()([branch1_dropout, branch2_dropout, branch3_dropout])

    # Fully connected layers
    flatten = Flatten()(merged)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model