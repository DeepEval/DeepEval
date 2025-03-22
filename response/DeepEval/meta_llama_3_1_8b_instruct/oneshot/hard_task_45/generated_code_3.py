import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Concatenate, BatchNormalization, Flatten, Dense, Lambda

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First block: Split the input into three groups and apply depthwise separable convolutional layers
    group1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1)[0])(input_layer)
    conv1_group1 = layers.SeparableConv2D(32, kernel_size=(1, 1), activation='relu')(group1)
    
    group2 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1)[1])(input_layer)
    conv1_group2 = layers.SeparableConv2D(32, kernel_size=(1, 1), activation='relu')(group2)
    
    group3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1)[2])(input_layer)
    conv1_group3 = layers.SeparableConv2D(32, kernel_size=(1, 1), activation='relu')(group3)
    
    # Concatenate the outputs from the three groups
    output_block1 = Concatenate()([conv1_group1, conv1_group2, conv1_group3])
    
    # Second block: Multiple branches for feature extraction
    branch1 = layers.SeparableConv2D(32, kernel_size=(1, 1), activation='relu')(output_block1)
    branch2 = layers.SeparableConv2D(32, kernel_size=(1, 1), activation='relu')(output_block1)
    branch2 = layers.SeparableConv2D(32, kernel_size=(3, 3), activation='relu')(branch2)
    
    branch3 = layers.SeparableConv2D(32, kernel_size=(1, 1), activation='relu')(output_block1)
    branch3 = layers.SeparableConv2D(32, kernel_size=(3, 3), activation='relu')(branch3)
    
    branch4 = layers.MaxPooling2D(pool_size=(2, 2))(output_block1)
    branch4 = layers.SeparableConv2D(32, kernel_size=(1, 1), activation='relu')(branch4)
    
    # Concatenate the outputs from all branches
    output_block2 = Concatenate()([branch1, branch2, branch3, branch4])
    
    # Apply batch normalization and flatten the result
    bath_norm = layers.BatchNormalization()(output_block2)
    flatten_layer = layers.Flatten()(bath_norm)
    
    # Output layer
    output_layer = layers.Dense(10, activation='softmax')(flatten_layer)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model