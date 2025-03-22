import keras
from keras.layers import Input, Lambda, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense
from keras.models import Model
import tensorflow as tf

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Process each group with a depthwise separable convolutional layer
    def depthwise_separable_layer(input_tensor, kernel_size):
        conv = Conv2D(filters=32, kernel_size=kernel_size, activation='relu')(input_tensor)
        conv = Conv2D(filters=32, kernel_size=kernel_size, activation='relu')(conv)
        conv = Conv2D(filters=num_classes, kernel_size=kernel_size, padding='same')(conv)
        return conv
    
    # Apply depthwise separable layers to each group
    group1 = depthwise_separable_layer(input=split_layer[0], kernel_size=(1, 1))
    group2 = depthwise_separable_layer(input=split_layer[1], kernel_size=(3, 3))
    group3 = depthwise_separable_layer(input=split_layer[2], kernel_size=(5, 5))
    
    # Concatenate the outputs of the three groups
    concatenated = Concatenate()(inputs=[group1, group2, group3])
    
    # Flatten the concatenated output and pass through a fully connected layer
    flattened = Flatten()(concatenated)
    output_layer = Dense(units=num_classes, activation='softmax')(flattened)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Number of classes in CIFAR-10 (10 for the 10 categories)
num_classes = 10

# Build the model
model = dl_model()

# Summary of the model
model.summary()