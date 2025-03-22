import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Lambda, BatchNormalization, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Define feature extraction for each group using depthwise separable convolutions
    def depthwise_separable_conv(input_tensor, kernel_size):
        conv = Conv2D(filters=None, kernel_size=kernel_size, padding='same', depthwise_constraint=None, pointwise_constraint=None)(input_tensor)
        bn = BatchNormalization()(conv)
        activation = tf.nn.relu(bn)
        return activation
    
    # Apply depthwise separable convolutions with different kernel sizes
    outputs = [depthwise_separable_conv(group, kernel_size) for group, kernel_size in zip(split_layer, [1, 3, 5])]
    
    # Concatenate the outputs of the three groups
    concatenated = Concatenate()(outputs)
    
    # Flatten the concatenated features
    flattened = Flatten()(concatenated)
    
    # Fully connected layer for classification
    dense_layer = Dense(units=10, activation='softmax')(flattened)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=dense_layer)
    
    return model