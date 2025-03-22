import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Add
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    def first_block(input_tensor):
        # Main path
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        
        # Branch path
        branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Combine paths
        output_tensor = Add()([conv2, branch])
        return output_tensor

    first_block_output = first_block(input_layer)
    batch_norm1 = BatchNormalization()(first_block_output)
    flatten_layer = Flatten()(batch_norm1)

    # Second block
    def second_block(input_tensor):
        # Split the input into three groups
        split_layer = Lambda(lambda x: keras.backend.split(x, num_or_size_splits=3, axis=-1))(input_tensor)

        # Extract features using depthwise separable convolutional layers
        def depthwise_conv_block(input_tensor, kernel_size):
            return Conv2D(filters=None, kernel_size=kernel_size, strides=(1, 1), padding='same', activation='relu', depthwise_mode=True)(input_tensor)

        path1 = depthwise_conv_block(split_layer[0], (1, 1))
        path2 = depthwise_conv_block(split_layer[1], (3, 3))
        path3 = depthwise_conv_block(split_layer[2], (5, 5))

        # Concatenate the outputs
        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor

    second_block_output = second_block(flatten_layer)
    batch_norm2 = BatchNormalization()(second_block_output)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(batch_norm2)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model