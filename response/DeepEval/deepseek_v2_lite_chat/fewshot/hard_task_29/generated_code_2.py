import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    # Block 1
    def block1(x):
        # Main path
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        # Branch path
        branch_conv = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(x)
        pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)
        # Add main and branch path outputs
        add_layer = Add()([conv2, branch_conv])
        # Flatten and pass through fully connected layer
        flat = Flatten()(add_layer)
        dense1 = Dense(units=128, activation='relu')(flat)

        return dense1

    # Block 2
    def block2(x):
        # Max pooling layers with varying scales
        pool1 = MaxPooling2D(pool_size=(1, 1), padding='same')(x)
        pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        pool3 = MaxPooling2D(pool_size=(4, 4), padding='same')(x)
        # Flatten and concatenate
        concat = Flatten()(keras.layers.concatenate([pool1, pool2, pool3]))
        # Pass through fully connected layers
        dense1 = Dense(units=128, activation='relu')(concat)
        dense2 = Dense(units=64, activation='relu')(dense1)
        output = Dense(units=10, activation='softmax')(dense2)

        return output

    # Input layer
    inputs = Input(shape=(28, 28, 1))

    # Apply Block 1
    block1_output = block1(inputs)

    # Apply Block 2
    block2_output = block2(block1_output)

    # Construct model
    model = keras.Model(inputs=inputs, outputs=block2_output)

    return model