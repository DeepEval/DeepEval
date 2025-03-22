import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main path
    def main_path(input_tensor):
        # Split the input into 3 groups
        split_tensors = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        group1 = split_tensors[0]  # First group remains unchanged
        group2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensors[1])  # Second group with convolution
        group3 = split_tensors[2]  # Third group remains unchanged
        
        # Combine output of group2 with group3
        combined = Add()([group2, group3])
        
        # Pass through an additional 3x3 convolution
        conv_output = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(combined)
        
        # Concatenate all groups
        output_tensor = Concatenate()([group1, conv_output, group3])
        return output_tensor

    # Branch path
    def branch_path(input_tensor):
        return Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)

    main_output = main_path(input_layer)
    branch_output = branch_path(input_layer)

    # Fuse both paths
    combined_output = Add()([main_output, branch_output])

    # Flatten and fully connected layer for classification
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model