import keras
from keras.layers import Input, Conv2D, Add, Lambda, SeparableConv2D, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First Block
    def first_block(input_tensor):
        # Main path
        main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
        main_path = Conv2D(filters=input_tensor.shape[-1], kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
        
        # Branch path
        branch_path = input_tensor  # Direct connection

        # Combine paths
        output_tensor = Add()([main_path, branch_path])
        return output_tensor

    # Second Block
    def second_block(input_tensor):
        # Split the input into three groups along the channel dimension
        split_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Each group goes through depthwise separable convolutions
        group1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_groups[0])
        group2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_groups[1])
        group3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_groups[2])
        
        # Concatenate the outputs
        output_tensor = Concatenate()([group1, group2, group3])
        return output_tensor

    # Process through the two blocks
    block1_output = first_block(input_layer)
    block2_output = second_block(block1_output)

    # Flatten and fully connected layers
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model