import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Lambda, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    block_1_inputs = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)
    block_1_outputs = []
    for i in range(3):
        block_1_outputs.append(Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block_1_inputs[i]))
    block_1_output = Add()(block_1_outputs)

    # Block 2
    block_2_input = Reshape(target_shape=(32, 32, 3, 1))(block_1_output)
    block_2_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block_2_input)
    block_2_output = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(block_2_output)
    block_2_output = Reshape(target_shape=(16, 16, 1))(block_2_output)

    # Block 3
    block_3_input = Reshape(target_shape=(16, 16, 1, 1))(block_2_output)
    block_3_output = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block_3_input)
    block_3_output = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(block_3_output)
    block_3_output = Reshape(target_shape=(8, 8, 1))(block_3_output)

    # Branch path
    branch_path_input = Reshape(target_shape=(8, 8, 1, 1))(input_layer)
    branch_path_output = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch_path_input)

    # Combine outputs from main path and branch path
    combined_outputs = Add()([block_1_output, branch_path_output])

    # Flatten and pass through fully connected layer
    flattened_output = Flatten()(combined_outputs)
    output_layer = Dense(units=10, activation='softmax')(flattened_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model