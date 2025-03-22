import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Lambda, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Apply 1x1 convolution to each group
    conv_groups = [Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(group) for group in split_layer]
    
    # Perform average pooling on each group
    pooled_groups = [AveragePooling2D(pool_size=(2, 2), strides=2)(group) for group in conv_groups]
    
    # Concatenate the three groups of feature maps along the channel dimension
    concatenated_features = keras.layers.concatenate(pooled_groups, axis=-1)
    
    # Flatten the concatenated feature maps into a one-dimensional vector
    flattened_features = Flatten()(concatenated_features)
    
    # Pass the flattened vector through two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flattened_features)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model