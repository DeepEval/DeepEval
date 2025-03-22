import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Lambda
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Apply 1x1 convolution to each group to reduce the number of channels
    conv_groups = [Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(group) for group in split_layer]
    
    # Apply average pooling to each group
    pooled_groups = [AveragePooling2D(pool_size=(8, 8))(group) for group in conv_groups]
    
    # Concatenate the three groups along the channel dimension
    concatenated = Concatenate(axis=-1)(pooled_groups)
    
    # Flatten the concatenated feature maps
    flattened = Flatten()(concatenated)
    
    # Pass through two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model