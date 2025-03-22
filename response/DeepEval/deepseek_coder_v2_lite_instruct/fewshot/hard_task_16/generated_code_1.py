import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dense, Lambda, Reshape, Add, Flatten
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    def block_1(input_tensor):
        # Split the input into three groups
        split_groups = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)
        
        # Process each group with a series of convolutions
        conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_groups[0])
        conv3_3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1_1)
        conv1_2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(conv3_3)
        
        conv1_3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_groups[1])
        conv3_4 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1_3)
        conv1_4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(conv3_4)
        
        conv1_5 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_groups[2])
        conv3_5 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1_5)
        conv1_6 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(conv3_5)
        
        # Concatenate the outputs from the three groups
        concatenated = Concatenate(axis=-1)([conv1_2, conv1_4, conv1_6])
        return concatenated

    block1_output = block_1(input_layer)

    # Transition Convolution to match the number of channels with the input
    transition_conv = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(block1_output)

    def block_2(input_tensor):
        # Global Max Pooling
        global_max_pool = MaxPooling2D(pool_size=(8, 8), strides=8)(input_tensor)
        
        # Generate weights through fully connected layers
        fc1 = Dense(units=128, activation='relu')(global_max_pool)
        fc2 = Dense(units=64, activation='relu')(fc1)
        
        # Reshape weights to match the shape of the input
        reshaped_weights = Reshape(target_shape=(1, 1, 64))(fc2)
        
        # Multiply weights with the input
        main_path_output = tf.multiply(input_tensor, reshaped_weights)
        
        return main_path_output

    block2_output = block_2(transition_conv)

    # Branch connecting directly to the input
    branch = input_layer

    # Add the outputs from both the main path and the branch
    added_output = Add()([block2_output, branch])

    # Flatten the final output and pass it through a fully connected layer for classification
    flattened = Flatten()(added_output)
    output_layer = Dense(units=10, activation='softmax')(flattened)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model