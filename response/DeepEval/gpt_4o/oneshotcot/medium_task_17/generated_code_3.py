import keras
from keras.layers import Input, Reshape, Permute, Dense, Flatten
from keras.models import Model
import tensorflow as tf

def dl_model():
    input_shape = (32, 32, 3)
    num_classes = 10
    groups = 3

    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Calculate channels per group
    channels_per_group = input_shape[-1] // groups

    # Step 1: Reshape the input to (height, width, groups, channels_per_group)
    reshape1 = Reshape((input_shape[0], input_shape[1], groups, channels_per_group))(input_layer)

    # Step 2: Permute the dimensions to swap the groups and channels_per_group
    permute = Permute((1, 2, 4, 3))(reshape1)

    # Step 3: Reshape back to the original shape
    reshape2 = Reshape(input_shape)(permute)

    # Step 4: Flatten and add a dense layer with softmax activation
    flatten = Flatten()(reshape2)
    output_layer = Dense(units=num_classes, activation='softmax')(flatten)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model