import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, DepthwiseConv2D, Concatenate, Dense, Flatten, Add
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Main path
    # Group 1: 1x1 Depthwise Separable Convolution
    group1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same')(split_inputs[0])
    group1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(group1)
    
    # Group 2: 3x3 Depthwise Separable Convolution
    group2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same')(split_inputs[1])
    group2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(group2)

    # Group 3: 5x5 Depthwise Separable Convolution
    group3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same')(split_inputs[2])
    group3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(group3)

    # Concatenate the outputs of the three groups
    main_path_output = Concatenate(axis=-1)([group1, group2, group3])

    # Branch path: 1x1 Convolution to align number of channels
    branch_path_output = Conv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Add main and branch path outputs
    combined_output = Add()([main_path_output, branch_path_output])

    # Flatten and fully connected layers
    flatten_output = Flatten()(combined_output)
    fc1 = Dense(128, activation='relu')(flatten_output)
    output_layer = Dense(10, activation='softmax')(fc1)  # 10 classes for CIFAR-10

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of using the model
if __name__ == "__main__":
    model = dl_model()
    model.summary()