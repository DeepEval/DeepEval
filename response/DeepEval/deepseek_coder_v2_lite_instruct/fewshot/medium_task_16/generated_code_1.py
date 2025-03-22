import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Lambda, Flatten, Dense, Concatenate
import tensorflow as tf

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Apply 1x1 convolutions to each group
    conv_groups = [Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split) for split in split_layer]
    
    # Perform average pooling on each group
    pooled_groups = [AveragePooling2D(pool_size=(8, 8), strides=(8, 8), padding='same')(group) for group in conv_groups]
    
    # Concatenate the three groups along the channel dimension
    concatenated_features = Concatenate(axis=-1)(pooled_groups)
    
    # Flatten the concatenated feature maps
    flattened_features = Flatten()(concatenated_features)
    
    # Pass through two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flattened_features)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model