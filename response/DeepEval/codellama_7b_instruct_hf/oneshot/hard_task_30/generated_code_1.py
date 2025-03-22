import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.layers import Lambda, Split
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First block
    main_path = input_layer
    branch_path = input_layer

    # Main path
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)

    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_path)

    # Combine main and branch paths
    output_layer = Add()([main_path, branch_path])

    # Second block
    input_layer = output_layer

    # Split input into three groups
    input_layer = Split(axis=-1, num_or_size_splits=3)(input_layer)

    # Extract features using depthwise separable convolutional layers
    groups = []
    for i in range(3):
        groups.append(Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer[i]))

    # Concatenate outputs from each group
    output_layer = Concatenate()(groups)

    # Flatten output layer
    output_layer = Flatten()(output_layer)

    # Add fully connected layers
    output_layer = Dense(units=128, activation='relu')(output_layer)
    output_layer = Dense(units=64, activation='relu')(output_layer)
    output_layer = Dense(units=10, activation='softmax')(output_layer)

    # Create and return model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model