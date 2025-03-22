import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    def main_path(input_tensor):
        # Split input into 3 groups
        inputs_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Feature extraction with different kernel sizes
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        
        # Concatenate the outputs
        output_tensor = Concatenate()([conv1, conv2, conv3])
        
        return output_tensor

    def branch_path(input_tensor):
        # 1x1 convolution to align output channels
        conv = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv

    # Process input through main and branch paths
    main_output = main_path(input_layer)
    branch_output = branch_path(input_layer)

    # Combine main and branch paths using addition
    fused_features = Add()([main_output, branch_output])

    # Classification using fully connected layers
    flatten_layer = Flatten()(fused_features)
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model