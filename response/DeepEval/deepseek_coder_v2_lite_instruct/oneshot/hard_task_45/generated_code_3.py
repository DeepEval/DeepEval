import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Add
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First Block
    def first_block(input_tensor):
        # Depthwise separable convolutions
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(input_tensor)
        
        # Concatenate outputs
        output_tensor = Concatenate()([conv1x1, conv3x3, conv5x5])
        return output_tensor

    first_block_output = first_block(input_layer)

    # Second Block
    def second_block(input_tensor):
        # Define branches
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
        branch3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch3)
        branch4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(input_tensor)
        branch4 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(branch4)

        # Concatenate outputs
        output_tensor = Concatenate()([branch1, branch2, branch3, branch4])
        return output_tensor

    second_block_output = second_block(first_block_output)

    # Batch Normalization and Flattening
    batch_norm = BatchNormalization()(second_block_output)
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Create the model
model = dl_model()
model.summary()