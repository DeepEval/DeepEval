import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Concatenate, Lambda, Reshape
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block_1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), activation='relu')(inputs_groups[1])
        conv3 = Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), activation='relu')(inputs_groups[2])
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    # Block 2
    def block_2(input_tensor):
        shape = tf.shape(input_tensor)
        reshaped = Reshape(target_shape=(shape[1], shape[2], 3, shape[3] // 3))(input_tensor)
        permuted = tf.transpose(reshaped, perm=[0, 1, 3, 2])  # Swap groups and channels
        output_tensor = Reshape(target_shape=(shape[1], shape[2], shape[3]))(permuted)
        return output_tensor

    # Block 3
    def block_3(input_tensor):
        output_tensor = Conv2D(filters=input_tensor.shape[-1], kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        return output_tensor

    # Main path
    block1_output1 = block_1(input_layer)
    block2_output = block_2(block1_output1)
    block3_output = block_3(block2_output)
    
    block1_output2 = block_1(block3_output)

    # Branch path
    branch_path = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)

    # Concatenate outputs from the main path and branch path
    combined_output = Concatenate()([block1_output2, branch_path])

    # Final classification layer
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create and return the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model