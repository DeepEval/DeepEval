import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, SeparableConv2D, Lambda
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    def first_block(input_tensor):
        # Split the input into three groups along the channel
        split_1 = Lambda(lambda x: x[:, :, :, :128])(input_tensor)
        split_2 = Lambda(lambda x: x[:, :, :, 128:256])(input_tensor)
        split_3 = Lambda(lambda x: x[:, :, :, 256:])(input_tensor)

        # Separate convolutions for each group
        conv_1x1 = Conv2D(64, (1, 1), activation='relu')(split_1)
        conv_3x3 = Conv2D(64, (3, 3), activation='relu')(split_2)
        conv_5x5 = Conv2D(64, (5, 5), activation='relu')(split_3)

        # Concatenate the outputs
        concatenated = Concatenate()([conv_1x1, conv_3x3, conv_5x5])
        return concatenated

    first_block_output = first_block(input_layer)
    batch_norm_1 = BatchNormalization()(first_block_output)

    # Second block
    def second_block(input_tensor):
        # Branch 1: 3x3 convolution
        branch_1 = Conv2D(64, (3, 3), activation='relu')(input_tensor)

        # Branch 2: 1x1 followed by two 3x3 convolutions
        branch_2 = Conv2D(32, (1, 1), activation='relu')(input_tensor)
        branch_2 = Conv2D(32, (3, 3), activation='relu')(branch_2)
        branch_2 = Conv2D(32, (3, 3), activation='relu')(branch_2)

        # Branch 3: max pooling
        branch_3 = MaxPooling2D((3, 3))(input_tensor)

        # Concatenate the outputs
        concatenated = Concatenate()([branch_1, branch_2, branch_3])
        return concatenated

    second_block_output = second_block(batch_norm_1)
    batch_norm_2 = BatchNormalization()(second_block_output)

    # Global average pooling and fully connected layer
    gap = GlobalAveragePooling2D()(batch_norm_2)
    dense = Dense(10, activation='softmax')(gap)

    model = Model(inputs=input_layer, outputs=dense)
    return model

# Create the model
model = dl_model()
model.summary()