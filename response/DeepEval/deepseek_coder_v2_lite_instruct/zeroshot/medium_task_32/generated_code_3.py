import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Lambda, Flatten, Dense

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 image shape
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Split the input into three groups along the last dimension
    splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    # Process each group with depthwise separable convolutional layers
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(splits[0])
    conv3x3 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(splits[1])
    conv5x5 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(splits[2])
    
    # Concatenate the outputs
    concatenated = tf.concat([conv1x1, conv3x3, conv5x5], axis=-1)
    
    # Flatten the concatenated features
    flattened = Flatten()(concatenated)
    
    # Fully connected layer for classification
    outputs = Dense(units=10, activation='softmax')(flattened)
    
    # Build the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()