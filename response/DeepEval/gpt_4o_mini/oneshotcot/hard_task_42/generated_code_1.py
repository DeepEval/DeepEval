import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dropout, Dense, Conv2D, Concatenate, Reshape, AveragePooling2D, BatchNormalization
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    # Path 1: MaxPooling 1x1
    path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    path1 = Flatten()(path1)
    path1 = Dropout(0.2)(path1)

    # Path 2: MaxPooling 2x2
    path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    path2 = Flatten()(path2)
    path2 = Dropout(0.2)(path2)

    # Path 3: MaxPooling 4x4
    path3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)
    path3 = Flatten()(path3)
    path3 = Dropout(0.2)(path3)

    # Concatenate the outputs of all paths
    block1_output = Concatenate()([path1, path2, path3])

    # Fully connected layer before Block 2
    fc1 = Dense(units=128, activation='relu')(block1_output)
    reshaped = Reshape((1, 1, 128))(fc1)  # Reshape to a 4D tensor

    # Block 2
    # Path 1: 1x1 Convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(reshaped)

    # Path 2: 1x1 Convolution followed by 1x7 and 7x1 Convolutions
    path2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(reshaped)
    path2 = Conv2D(filters=32, kernel_size=(1, 7), padding='same', activation='relu')(path2)
    path2 = Conv2D(filters=32, kernel_size=(7, 1), padding='same', activation='relu')(path2)

    # Path 3: 1x1 Convolution followed by alternating 7x1 and 1x7 Convolutions
    path3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(reshaped)
    path3 = Conv2D(filters=32, kernel_size=(7, 1), padding='same', activation='relu')(path3)
    path3 = Conv2D(filters=32, kernel_size=(1, 7), padding='same', activation='relu')(path3)
    path3 = Conv2D(filters=32, kernel_size=(7, 1), padding='same', activation='relu')(path3)

    # Path 4: Average Pooling with a 1x1 Convolution
    path4 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(reshaped)
    path4 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(path4)

    # Concatenate the outputs of all paths in Block 2
    block2_output = Concatenate()([path1, path2, path3, path4])
    block2_output = Flatten()(block2_output)

    # Final classification layers
    dense1 = Dense(units=128, activation='relu')(block2_output)
    dense1 = Dropout(0.5)(dense1)  # Adding dropout for regularization
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model