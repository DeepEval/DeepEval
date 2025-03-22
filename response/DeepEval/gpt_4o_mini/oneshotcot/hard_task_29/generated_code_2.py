import keras
from keras.layers import Input, Conv2D, Add, MaxPooling2D, Flatten, Concatenate, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))  # Input layer for MNIST dataset

    # First block
    # Main path
    main_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_conv2 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_conv1)

    # Branch path
    branch_path = input_layer

    # Combine paths
    block1_output = Add()([main_conv2, branch_path])

    # Second block: Three MaxPooling layers with varying scales
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(block1_output)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(block1_output)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(block1_output)

    # Flatten the outputs
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)

    # Concatenate flattened outputs
    concatenated_output = Concatenate()([flatten1, flatten2, flatten3])

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(concatenated_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # 10 output classes for MNIST

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model