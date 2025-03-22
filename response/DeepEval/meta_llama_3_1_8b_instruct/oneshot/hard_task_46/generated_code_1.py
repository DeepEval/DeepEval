import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():     

    input_layer = keras.Input(shape=(32, 32, 3))
    
    # Block 1: Split the input into three groups and apply separable convolution
    def separable_convolution(input_tensor, kernel_size):
        return layers.SeparableConv2D(filters=32, kernel_size=kernel_size, padding='same')(input_tensor)
    
    split_layer = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    conv1x1 = separable_convolution(split_layer[0], (1, 1))
    conv3x3 = separable_convolution(split_layer[1], (3, 3))
    conv5x5 = separable_convolution(split_layer[2], (5, 5))
    concat_layer = layers.Concatenate()([conv1x1, conv3x3, conv5x5])

    # Block 2: Feature extraction with multiple branches
    conv3x3_branch = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(concat_layer)
    conv1x1_branch = layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same')(concat_layer)
    conv3x3_branch = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(conv3x3_branch)
    conv3x3_branch = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(conv3x3_branch)
    maxpool_branch = layers.MaxPooling2D(pool_size=(2, 2))(concat_layer)
    
    branch_concat = layers.Concatenate()([conv3x3_branch, conv1x1_branch, maxpool_branch])

    # Global average pooling and fully connected layer
    gap_layer = layers.GlobalAveragePooling2D()(branch_concat)
    output_layer = layers.Dense(units=10, activation='softmax')(gap_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model