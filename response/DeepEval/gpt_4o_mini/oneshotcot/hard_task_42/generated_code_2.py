import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dropout, Dense, Concatenate, Conv2D, Reshape
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    # Path 1: 1x1 MaxPooling
    path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    path1 = Flatten()(path1)
    path1 = Dropout(0.5)(path1)

    # Path 2: 2x2 MaxPooling
    path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    path2 = Flatten()(path2)
    path2 = Dropout(0.5)(path2)

    # Path 3: 4x4 MaxPooling
    path3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
    path3 = Flatten()(path3)
    path3 = Dropout(0.5)(path3)

    # Concatenate outputs from Block 1
    block1_output = Concatenate()([path1, path2, path3])

    # Fully connected layer after Block 1
    fc1 = Dense(units=128, activation='relu')(block1_output)

    # Reshape the output to prepare for Block 2
    reshaped = Reshape((1, 1, 128))(fc1)  # Reshape to (1, 1, 128) for the next block

    # Block 2
    # Path 1: 1x1 Convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(reshaped)

    # Path 2: 1x1 Conv -> 1x7 Conv -> 7x1 Conv
    path2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(reshaped)
    path2 = Conv2D(filters=32, kernel_size=(1, 7), padding='same', activation='relu')(path2)
    path2 = Conv2D(filters=32, kernel_size=(7, 1), padding='same', activation='relu')(path2)

    # Path 3: 1x1 Conv -> 7x1 Conv -> 1x7 Conv (alternating)
    path3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(reshaped)
    path3 = Conv2D(filters=32, kernel_size=(7, 1), padding='same', activation='relu')(path3)
    path3 = Conv2D(filters=32, kernel_size=(1, 7), padding='same', activation='relu')(path3)

    # Path 4: Average Pooling + 1x1 Conv
    path4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(reshaped)
    path4 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(path4)

    # Concatenate outputs from Block 2
    block2_output = Concatenate()([path1, path2, path3, path4])

    # Flatten the output from Block 2
    flat_output = Flatten()(block2_output)

    # Fully connected layers for final classification
    dense1 = Dense(units=64, activation='relu')(flat_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model