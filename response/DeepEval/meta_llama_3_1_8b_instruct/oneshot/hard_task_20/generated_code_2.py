import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input layer
    input_layer = keras.Input(shape=(32, 32, 3))
    
    # Split the input into three groups
    x = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Define the main path with three feature extraction groups
    conv1x1_group = layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
    conv3x3_group = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])
    conv5x5_group = layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[2])
    
    # Concatenate the outputs from the three groups
    concat_main = layers.Concatenate()([conv1x1_group, conv3x3_group, conv5x5_group])
    
    # Define the branch path with a 1x1 convolutional layer
    conv1x1_branch = layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Add the outputs of the main and branch paths
    fused_features = layers.Add()([concat_main, conv1x1_branch])
    
    # Perform global average pooling
    gap = layers.GlobalAveragePooling2D()(fused_features)
    
    # Define the output layers
    dense1 = layers.Dense(units=64, activation='relu')(gap)
    output_layer = layers.Dense(units=10, activation='softmax')(dense1)
    
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model