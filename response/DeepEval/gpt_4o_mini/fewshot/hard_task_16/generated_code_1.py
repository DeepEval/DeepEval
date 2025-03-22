import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Concatenate, GlobalMaxPooling2D, Dense, Add, Reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are typically 32x32 pixels with 3 color channels

    # Block 1
    def block_1(input_tensor):
        # Split input into three groups
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Group 1: 1x1 Conv
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        # Group 2: 3x3 Conv
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        # Group 3: 1x1 Conv
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        
        # Concatenate outputs of the three groups
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)

    # Transition Convolution
    transition_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1_output)

    # Block 2
    def block_2(input_tensor):
        # Global max pooling
        pooled_output = GlobalMaxPooling2D()(input_tensor)
        # Two fully connected layers to generate weights
        dense1 = Dense(units=64, activation='relu')(pooled_output)
        weights = Dense(units=input_tensor.shape[-1], activation='sigmoid')(dense1)  # Match the number of channels
        reshaped_weights = Reshape(target_shape=(1, 1, input_tensor.shape[-1]))(weights)  # Reshape to match the input
        return reshaped_weights

    weights = block_2(input_tensor=transition_conv)

    # Main path output
    main_path_output = tf.multiply(transition_conv, weights)  # Element-wise multiplication

    # Branch output (directly from the input layer)
    branch_output = input_layer

    # Combine both outputs
    combined_output = Add()([main_path_output, branch_output])

    # Final classification layer
    flatten = GlobalMaxPooling2D()(combined_output)  # Flatten the combined output
    output_layer = Dense(units=10, activation='softmax')(flatten)  # CIFAR-10 has 10 classes

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model