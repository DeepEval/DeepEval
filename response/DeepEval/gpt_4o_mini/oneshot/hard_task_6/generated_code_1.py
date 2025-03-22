import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Lambda, Reshape
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block1(input_tensor):
        split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv_outputs = [Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), activation='relu')(t) for t in split_tensors]
        return Concatenate()(conv_outputs)

    # Block 2
    def block2(input_tensor):
        batch_size, height, width, channels = tf.shape(input_tensor)[0], tf.shape(input_tensor)[1], tf.shape(input_tensor)[2], tf.shape(input_tensor)[3]
        reshaped_tensor = Reshape((height, width, 3, channels // 3))(input_tensor)
        permuted_tensor = Lambda(lambda x: tf.transpose(x, perm=[0, 1, 3, 2, 4]))(reshaped_tensor)
        return Reshape((height, width, channels))(permuted_tensor)

    # Block 3
    def block3(input_tensor):
        return Conv2D(filters=input_tensor.shape[-1], kernel_size=(3, 3), padding='same', activation='relu', depthwise=True)(input_tensor)

    # Main path
    block1_output = block1(input_layer)
    block2_output = block2(block1_output)
    block3_output = block3(block2_output)
    block1_output_repeat = block1(block3_output)
    
    # Branch path
    branch_output = AveragePooling2D(pool_size=(2, 2))(input_layer)

    # Concatenate main path and branch path
    concatenated_output = Concatenate()([block1_output_repeat, branch_output])
    
    # Fully connected layer
    flatten_layer = Flatten()(concatenated_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model