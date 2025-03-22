import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Lambda, Concatenate, AveragePooling2D
import tensorflow as tf

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define Block 1 for the main path
    def block_1(input_tensor):
        split_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_groups[0])
        conv1_2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_groups[1])
        conv1_3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_groups[2])
        output_tensor = Concatenate()([conv1_1, conv1_2, conv1_3])
        return output_tensor

    # Apply Block 1 three times for the main path
    main_path = block_1(input_layer)
    main_path = block_1(main_path)
    main_path = block_1(main_path)

    # Define Block 2 for the main path
    def block_2(input_tensor):
        shape = tf.keras.backend.int_shape(input_tensor)
        height, width, channels = shape[1], shape[2], shape[3]
        reshaped = tf.keras.backend.reshape(input_tensor, (height, width, 3, channels // 3))
        permuted = tf.keras.backend.permute_dimensions(reshaped, (0, 1, 3, 2))
        output_tensor = tf.keras.backend.reshape(permuted, (height, width, channels))
        return output_tensor

    main_path = block_2(main_path)

    # Define Block 3 for the main path
    main_path = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(main_path)

    # Define the branch path
    branch_path = AveragePooling2D(pool_size=(8, 8), strides=(8, 8), padding='same')(input_layer)
    branch_path = Flatten()(branch_path)

    # Concatenate the outputs from the main path and the branch path
    combined_output = Concatenate()([main_path, branch_path])

    # Pass the combined output through a fully connected layer
    output_layer = Dense(units=10, activation='softmax')(combined_output)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model