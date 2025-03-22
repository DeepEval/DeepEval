import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Lambda
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First Block
    def first_block(input_tensor):
        # Split the input into three groups along the channel axis
        split_1 = Lambda(lambda x: x[:, :, :, :128])(input_tensor)
        split_2 = Lambda(lambda x: x[:, :, :, 128:256])(input_tensor)
        split_3 = Lambda(lambda x: x[:, :, :, 256:])(input_tensor)

        # Separable convolutions
        conv_1x1 = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(split_1)
        conv_3x3 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', dilation_rate=(1, 1))(split_2)
        conv_5x5 = Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu', dilation_rate=(1, 1))(split_3)

        # Concatenate the outputs
        concatenated = Concatenate()([conv_1x1, conv_3x3, conv_5x5])
        return concatenated

    block_output = first_block(input_layer)
    batch_norm = BatchNormalization()(block_output)
    flatten_layer = Flatten()(batch_norm)
    dense1 = Dense(256, activation='relu')(flatten_layer)
    dense2 = Dense(128, activation='relu')(dense1)

    # Second Block
    def second_block(input_tensor):
        # First branch: 3x3 convolution
        branch1 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)

        # Second branch: 1x1 convolution followed by two 3x3 convolutions
        branch2 = Conv2D(32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(branch2)

        # Third branch: max pooling
        branch3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)

        # Concatenate the outputs
        concatenated = Concatenate()([branch1, branch2, branch3])
        return concatenated

    second_block_output = second_block(dense2)
    global_avg_pool = GlobalAveragePooling2D()(second_block_output)
    output_layer = Dense(10, activation='softmax')(global_avg_pool)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Create the model
model = dl_model()
model.summary()