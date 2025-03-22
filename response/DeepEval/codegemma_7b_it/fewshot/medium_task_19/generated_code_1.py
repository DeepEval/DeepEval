import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First branch: 1x1 convolution for dimensionality reduction
    branch_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Second branch: 1x1 convolution followed by 3x3 convolution
    branch_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_2)

    # Third branch: 1x1 convolution followed by 5x5 convolution
    branch_3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch_3)

    # Fourth branch: 3x3 max pooling followed by 1x1 convolution
    branch_4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    branch_4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_4)

    # Concatenate outputs from all branches
    merged_branches = Concatenate()([branch_1, branch_2, branch_3, branch_4])

    # Flatten features
    flatten_layer = Flatten()(merged_branches)

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model