import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Lambda, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Apply 1x1 convolutions to each group independently
    conv_groups = []
    for group in split_layer:
        conv_group = Conv2D(filters=group.shape[-1]//3, kernel_size=(1, 1), padding='same', activation='relu')(group)
        conv_groups.append(conv_group)
    
    # Downsample each group via average pooling
    pooled_groups = []
    for group in conv_groups:
        pooled_group = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(group)
        pooled_groups.append(pooled_group)
    
    # Concatenate the three groups of feature maps along the channel dimension
    concatenated_features = Concatenate(axis=-1)(pooled_groups)
    
    # Flatten the concatenated feature maps into a one-dimensional vector
    flattened_features = Flatten()(concatenated_features)
    
    # Pass through two fully connected layers for classification
    dense1 = Dense(units=256, activation='relu')(flattened_features)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model