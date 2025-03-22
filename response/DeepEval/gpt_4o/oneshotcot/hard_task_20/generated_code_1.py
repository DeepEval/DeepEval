import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Add, Flatten, Dense
from keras import backend as K

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have shape 32x32x3

    # Define the main path
    def main_path(input_tensor):
        # Split input into three parts along the channel axis
        split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)

        # Apply different convolution operations
        path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split[0])
        path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split[1])
        path3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split[2])

        # Concatenate the outputs
        main_output = Concatenate()([path1, path2, path3])
        return main_output

    # Define the branch path
    def branch_path(input_tensor):
        branch_output = Conv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        return branch_output

    # Get outputs from both paths
    main_output = main_path(input_layer)
    branch_output = branch_path(input_layer)

    # Fuse features by adding the outputs of the main and branch paths
    fused_features = Add()([main_output, branch_output])

    # Classification head
    flatten_layer = Flatten()(fused_features)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model