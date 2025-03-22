import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Dense, Flatten, Reshape, Lambda
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # First block
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=1)(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=2)(input_layer)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=4)(input_layer)
    
    # Flatten and concatenate the outputs of the first block
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)
    concat_layer = Concatenate()([flatten1, flatten2, flatten3])
    
    # Fully connected layer and reshape operation
    dense1 = Dense(units=128, activation='relu')(concat_layer)
    reshape_layer = Reshape((4, 4, 8))(dense1)  # Assuming the output of dense1 is reshaped to 4x4x8

    # Second block
    def depthwise_conv_block(inputs, kernel_size):
        depthwise_conv = Conv2D(filters=1, kernel_size=kernel_size, padding='same', depthwise_mode=True)(inputs)
        return depthwise_conv
    
    # Split the input into four groups and apply depthwise separable convolutions
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(reshape_layer)
    conv_groups = [depthwise_conv_block(split_layer[i], kernel_size=k) for i, k in enumerate([1, 3, 5, 7])]
    
    # Concatenate the outputs of the depthwise separable convolutional layers
    concat_conv_groups = Concatenate(axis=-1)(conv_groups)
    
    # Flatten the concatenated output
    flatten_conv_groups = Flatten()(concat_conv_groups)
    
    # Fully connected layer for classification
    dense2 = Dense(units=64, activation='relu')(flatten_conv_groups)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()