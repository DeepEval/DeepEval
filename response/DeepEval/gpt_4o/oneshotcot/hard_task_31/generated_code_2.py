import keras
from keras.layers import Input, Conv2D, Dropout, Add, Lambda, SeparableConv2D, Flatten, Dense, Concatenate
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block: Main path and branch path
    def first_block(input_tensor):
        # Main path
        main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        main_path = Dropout(rate=0.2)(main_path)
        main_path = Conv2D(filters=input_tensor.shape[-1], kernel_size=(3, 3), padding='same', activation='relu')(main_path)

        # Branch path
        branch_path = input_tensor

        # Add main path and branch path
        output_tensor = Add()([main_path, branch_path])
        return output_tensor

    first_block_output = first_block(input_layer)

    # Second block: Split and Separable Convolution
    def second_block(input_tensor):
        # Split into three groups
        split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)

        # Apply separable convolutions with different kernel sizes
        path1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_tensor[0])
        path1 = Dropout(rate=0.2)(path1)

        path2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_tensor[1])
        path2 = Dropout(rate=0.2)(path2)

        path3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_tensor[2])
        path3 = Dropout(rate=0.2)(path3)

        # Concatenate the outputs of the three paths
        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor

    second_block_output = second_block(first_block_output)

    # Flatten and Fully Connected Layer for classification
    flatten_layer = Flatten()(second_block_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model