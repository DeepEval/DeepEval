import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Dense, GlobalMaxPooling2D, Reshape, Multiply, Add, Flatten
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block1(input_tensor):
        # Split input into 3 along the last dimension
        split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Each split goes through a series of convolutions
        path1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_layer[0])
        path1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(path1)
        path1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(path1)

        path2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_layer[1])
        path2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(path2)
        path2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(path2)

        path3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_layer[2])
        path3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(path3)
        path3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(path3)

        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor
    
    block1_output = block1(input_layer)

    # Transition Convolution
    transition_conv = Conv2D(filters=3, kernel_size=(1, 1), activation='relu')(block1_output)

    # Block 2
    def block2(input_tensor):
        # Global max pooling
        global_max_pool = GlobalMaxPooling2D()(input_tensor)

        # Channel-matching weights through fully connected layers
        dense1 = Dense(units=16, activation='relu')(global_max_pool)
        weights = Dense(units=3, activation='sigmoid')(dense1)

        # Reshape weights to match the shape of adjusted output
        reshaped_weights = Reshape((1, 1, 3))(weights)

        # Multiply weights with the adjusted output
        weighted_output = Multiply()([input_tensor, reshaped_weights])
        return weighted_output
    
    block2_output = block2(transition_conv)

    # Direct branch connection to input
    direct_branch = Conv2D(filters=3, kernel_size=(1, 1), activation='relu')(input_layer)

    # Combine main path and branch
    combined_output = Add()([block2_output, direct_branch])

    # Final classification layer
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model