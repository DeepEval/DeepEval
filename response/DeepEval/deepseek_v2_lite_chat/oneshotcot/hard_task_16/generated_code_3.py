import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    def conv_block(input_tensor, num_filters):
        # Convolutional block with 1x1, 3x3, and 1x1 convolutions
        conv1 = Conv2D(num_filters, (1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(num_filters, (1, 1), padding='same', activation='relu')(conv2)
        return conv3

    def transition_layer(input_tensor, num_filters):
        # Transition layer to adjust channel count
        pool = MaxPooling2D(pool_size=(2, 2))(input_tensor)
        conv = Conv2D(num_filters, (1, 1), padding='same', activation='relu')(pool)
        return Conv2D(num_filters, (1, 1), padding='same', activation='relu')(conv)

    def block_1(input_tensor):
        # Split input into three groups
        group1 = Lambda(lambda x: keras.layers.split(x, 3, axis=-1))(input_tensor)
        group2 = Lambda(lambda x: keras.layers.split(x, 3, axis=-1))(input_tensor)
        group3 = Lambda(lambda x: keras.layers.split(x, 3, axis=-1))(input_tensor)

        # Convolutional blocks for each group
        c1 = conv_block(group1[0], 32)
        c2 = conv_block(group2[1], 32)
        c3 = conv_block(group3[2], 32)

        # Concatenate outputs from different paths
        concat = Concatenate()(c1 + c2 + c3)

        # Continue with batch normalization, flattening, and dense layers
        batch_norm = BatchNormalization()(concat)
        flat = Flatten()(batch_norm)
        dense1 = Dense(128, activation='relu')(flat)
        dense2 = Dense(64, activation='relu')(dense1)
        output = Dense(10, activation='softmax')(dense2)

        return output

    def block_2(input_tensor):
        # Global max pooling and dense layers
        pool = MaxPooling2D(pool_size=(8, 8))(input_tensor)
        reshape = Dense(128)(pool)
        reshape2 = Dense(64)(reshape)
        output = Dense(10, activation='softmax')(reshape2)

        return output

    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # First block
    block1 = block_1(inputs)
    transition = transition_layer(inputs, 64)

    # Second block
    block2 = block_2(transition)

    # Add branch directly from the input
    branch = Dense(10, activation='softmax')(inputs)

    # Combine outputs from main path and branch
    combined = keras.layers.Add()([block1, block2, branch])

    # Final dense layers
    dense = Dense(128, activation='relu')(combined)
    output = Dense(10, activation='softmax')(dense)

    # Model configuration
    model = keras.Model(inputs=inputs, outputs=output)

    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])