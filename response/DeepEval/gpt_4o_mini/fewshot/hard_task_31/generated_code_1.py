import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense, Lambda, Concatenate, SeparableConv2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block_1(input_tensor):
        # Main path
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        dropout1 = Dropout(0.3)(conv1)
        conv2 = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu')(dropout1)

        # Branch path
        branch_path = input_tensor

        # Add both paths
        output_tensor = Add()([conv2, branch_path])
        return output_tensor

    # Block 2
    def block_2(input_tensor):
        # Split the input into three groups
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)

        # Each group processes with separable convolutions
        conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        dropout1 = Dropout(0.3)(conv1)

        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(inputs_groups[1])
        dropout2 = Dropout(0.3)(conv2)

        conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(inputs_groups[2])
        dropout3 = Dropout(0.3)(conv3)

        # Concatenate the outputs
        output_tensor = Concatenate()([dropout1, dropout2, dropout3])
        return output_tensor

    # Process through the two blocks
    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    # Final layers
    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model