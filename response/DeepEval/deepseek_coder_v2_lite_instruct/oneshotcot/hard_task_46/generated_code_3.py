import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, AveragePooling2D
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    def first_block(input_tensor):
        # Split the input into three groups along the channel axis
        split_1 = Lambda(lambda x: x[:, :, :, :10])(input_tensor)
        split_2 = Lambda(lambda x: x[:, :, :, 10:20])(input_tensor)
        split_3 = Lambda(lambda x: x[:, :, :, 20:])(input_tensor)

        # Separate convolutional layers for each group
        conv_1x1 = Conv2D(10, (1, 1), activation='relu')(split_1)
        conv_3x3 = Conv2D(10, (3, 3), activation='relu', padding='same')(split_2)
        conv_5x5 = Conv2D(10, (5, 5), activation='relu', padding='same')(split_3)

        # Concatenate the outputs
        concatenated = Concatenate()([conv_1x1, conv_3x3, conv_5x5])
        return concatenated

    block_output = first_block(input_layer)
    batch_norm = BatchNormalization()(block_output)
    flatten_layer = Flatten()(batch_norm)

    # Second block
    def second_block(input_tensor):
        # 3x3 convolution branch
        conv_3x3 = Conv2D(20, (3, 3), activation='relu', padding='same')(input_tensor)

        # 1x1 -> 3x3 -> 3x3 branch
        conv_1x1 = Conv2D(10, (1, 1), activation='relu')(input_tensor)
        conv_3x3_1 = Conv2D(10, (3, 3), activation='relu', padding='same')(conv_1x1)
        conv_3x3_2 = Conv2D(10, (3, 3), activation='relu', padding='same')(conv_3x3_1)

        # Max pooling branch
        max_pooling = MaxPooling2D((3, 3), strides=(1, 1))(input_tensor)

        # Concatenate the outputs
        concatenated = Concatenate()([conv_3x3, conv_3x3_2, max_pooling])
        return concatenated

    second_block_output = second_block(flatten_layer)
    batch_norm_2 = BatchNormalization()(second_block_output)
    global_avg_pool = AveragePooling2D((8, 8))(batch_norm_2)
    dense_layer = Dense(10, activation='softmax')(global_avg_pool)

    model = Model(inputs=input_layer, outputs=dense_layer)
    return model

# Build the model
model = dl_model()
model.summary()