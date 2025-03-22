import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the basic block
    def basic_block(input_tensor):
        # Main path
        conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        relu = ReLU()(batch_norm)
        
        # Branch path
        branch = input_tensor
        
        # Addition of both paths
        output_tensor = Add()([relu, branch])
        return output_tensor

    # Initial convolutional layer to reduce dimensionality
    initial_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    batch_norm_initial = BatchNormalization()(initial_conv)
    relu_initial = ReLU()(batch_norm_initial)

    # First block with two basic blocks
    block1_output = basic_block(relu_initial)
    block1_output = basic_block(block1_output)

    # Second block with another basic block
    block2_output = basic_block(block1_output)

    # Average pooling layer to downsample the feature map
    avg_pool = AveragePooling2D(pool_size=(4, 4), strides=(1, 1))(block2_output)

    # Flatten the feature map
    flatten = Flatten()(avg_pool)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model