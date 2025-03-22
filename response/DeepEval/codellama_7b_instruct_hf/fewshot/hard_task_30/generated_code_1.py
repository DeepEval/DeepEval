import keras
from keras.layers import Input, Dense, Flatten, Concatenate, Conv2D, MaxPooling2D, AveragePooling2D, Lambda, DepthwiseConv2D, Reshape

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Dual-path structure
    main_path = Conv2D(32, (3, 3), activation='relu')(input_layer)
    main_path = Conv2D(32, (3, 3), activation='relu')(main_path)
    main_path = MaxPooling2D((2, 2))(main_path)
    branch_path = Conv2D(32, (1, 1), activation='relu')(input_layer)

    # Combine the main and branch paths
    output_layer = Add()([main_path, branch_path])

    # Split the output layer into three groups
    groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(output_layer)

    # Extract features using depthwise separable convolutional layers with different kernel sizes
    group1 = DepthwiseConv2D(kernel_size=(1, 1), activation='relu')(groups[0])
    group2 = DepthwiseConv2D(kernel_size=(3, 3), activation='relu')(groups[1])
    group3 = DepthwiseConv2D(kernel_size=(5, 5), activation='relu')(groups[2])

    # Concatenate the output from each group
    output_layer = Concatenate()([group1, group2, group3])

    # Flatten and add fully connected layers
    output_layer = Flatten()(output_layer)
    output_layer = Dense(64, activation='relu')(output_layer)
    output_layer = Dense(10, activation='softmax')(output_layer)

    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model