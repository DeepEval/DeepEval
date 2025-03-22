import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Concatenate, Flatten, Dense, MaxPooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Basic block
    def basic_block(input_tensor):
        conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        bn = BatchNormalization()(conv)
        relu = Activation('relu')(bn)
        # Branching out from the same input tensor
        branch = Concatenate(name='branch_A')(inputs=[input_tensor, relu])

        return branch

    # Feature extraction block
    def feature_extraction_block(input_tensor):
        block_output = basic_block(input_tensor)
        # Max pooling layer to downsample the feature map
        max_pool = MaxPooling2D(pool_size=(2, 2))(block_output)
        return max_pool

    # First block
    block1 = feature_extraction_block(input_layer)

    # Second block
    block2 = feature_extraction_block(block1)

    # Concatenate feature maps from both paths
    concat = Concatenate(name='concat_A')([block1, block2])

    # Add another convolutional layer to increase features
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(concat)
    bn2 = BatchNormalization()(conv2)
    relu2 = Activation('relu')(bn2)

    # Flatten and pass through fully connected layers
    flatten = Flatten(name='flatten_A')(relu2)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model