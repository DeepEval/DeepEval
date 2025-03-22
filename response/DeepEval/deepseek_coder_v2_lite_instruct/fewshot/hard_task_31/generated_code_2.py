import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Dropout, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    def block_1(input_tensor):
        # Main path
        conv_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_main)
        
        # Branch path
        conv_branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Addition of main and branch paths
        added = Add()([conv_main, conv_branch])
        return added

    def block_2(input_tensor):
        # Split the input into three groups
        split_layers = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Process each group with different kernel sizes
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layers[0])
        conv1 = Dropout(0.2)(conv1)
        
        conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layers[1])
        conv3 = Dropout(0.2)(conv3)
        
        conv5 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_layers[2])
        conv5 = Dropout(0.2)(conv5)
        
        # Concatenate the outputs
        concatenated = tf.concat([conv1, conv3, conv5], axis=-1)
        return concatenated

    # Apply blocks to input layer
    block1_output = block_1(input_layer)
    block2_output = block_2(block1_output)

    # Flatten and fully connected layer for classification
    flattened = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flattened)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model