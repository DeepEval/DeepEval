import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution and 3x3 convolution
    branch_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_1)
    branch_1 = Dropout(rate=0.2)(branch_1)

    # Branch 2: 1x1 convolution, 1x7 convolution, 7x1 convolution, and 3x3 convolution
    branch_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch_2)
    branch_2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch_2)
    branch_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_2)
    branch_2 = Dropout(rate=0.2)(branch_2)

    # Branch 3: max pooling
    branch_3 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(input_layer)
    branch_3 = Dropout(rate=0.2)(branch_3)

    # Concatenate branches
    merged_branches = keras.layers.concatenate([branch_1, branch_2, branch_3], axis=1)

    # Flatten and fully connected layers
    flattened = Flatten()(merged_branches)
    dense_1 = Dense(units=128, activation='relu')(flattened)
    dense_2 = Dense(units=64, activation='relu')(dense_1)
    output_layer = Dense(units=10, activation='softmax')(dense_2)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model