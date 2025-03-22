import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():

    input_layer = layers.Input(shape=(32, 32, 3))
    
    # Split the channels into three groups
    split_layer = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=2))(input_layer)
    
    # Apply different convolutional kernels to each group
    conv1 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_layer[0])
    conv2 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(split_layer[1])
    conv3 = layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(split_layer[2])
    
    # Concatenate the outputs from the three groups
    concat_layer = layers.Concatenate(axis=2)([conv1, conv2, conv3])
    
    # Flatten the output and pass through fully connected layers
    flatten_layer = layers.Flatten()(concat_layer)
    dense1 = layers.Dense(units=128, activation='relu')(flatten_layer)
    output_layer = layers.Dense(units=10, activation='softmax')(dense1)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    
    return model