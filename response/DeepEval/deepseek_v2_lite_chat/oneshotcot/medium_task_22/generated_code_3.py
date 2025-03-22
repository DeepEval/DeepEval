import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # Adjusted shape for CIFAR-10

    # Branch 1: 3x3 convolutions
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    branch1 = BatchNormalization()(branch1)
    branch1 = MaxPooling2D(pool_size=(2, 2))(branch1)

    # Branch 2: 1x1 convolution, followed by two 3x3 convolutions
    branch2 = Conv2D(filters=64, kernel_size=(1, 1))(input_layer)
    branch2 = BatchNormalization()(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = MaxPooling2D(pool_size=(2, 2))(branch2)

    # Branch 3: Max pooling
    branch3 = MaxPooling2D(pool_size=(2, 2))(input_layer)

    # Concatenate features from all branches
    concat = Concatenate()([])
    concat = Conv2D(filters=128, kernel_size=(1, 1), activation='relu')(concat)
    concat = BatchNormalization()(concat)

    # Flatten and pass through dense layers
    flatten = Flatten()(concat)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model