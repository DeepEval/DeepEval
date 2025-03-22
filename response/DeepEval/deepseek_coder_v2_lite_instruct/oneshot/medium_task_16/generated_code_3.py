import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Lambda, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Apply 1x1 convolution to each group independently
    conv_groups = []
    for group in split:
        conv = Conv2D(filters=group.shape[-1]//3, kernel_size=(1, 1), padding='same', activation='relu')(group)
        conv_groups.append(conv)
    
    # Downsample each group via average pooling
    pooled_groups = []
    for conv in conv_groups:
        pool = AveragePooling2D(pool_size=(2, 2), strides=2)(conv)
        pooled_groups.append(pool)
    
    # Concatenate the resulting groups along the channel dimension
    concatenated = Concatenate(axis=-1)(pooled_groups)
    
    # Flatten the concatenated feature maps
    flattened = Flatten()(concatenated)
    
    # Pass through two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model