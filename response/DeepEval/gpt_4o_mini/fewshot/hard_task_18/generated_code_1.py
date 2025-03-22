import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, AveragePooling2D, Add, GlobalAveragePooling2D, Dense, Reshape, Flatten

def dl_model():

    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # First block
    def block_1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
        return avg_pool

    block1_output = block_1(input_layer)

    # Adding the input to the output of the first block
    added_output = Add()([input_layer, block1_output])

    # Second block
    def block_2(input_tensor):
        # Global average pooling to generate channel weights
        global_avg_pool = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=32, activation='relu')(global_avg_pool)
        dense2 = Dense(units=32, activation='sigmoid')(dense1)
        
        # Reshape to create channel weights for each feature map
        reshaped_weights = Reshape((1, 1, 32))(dense2)
        
        # Multiply the input tensor by the channel weights
        weighted_output = tf.multiply(input_tensor, reshaped_weights)
        return weighted_output

    block2_output = block_2(added_output)

    # Flattening the output for classification
    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model