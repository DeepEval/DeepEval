import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Concatenate, Lambda

def dl_model():
    input_shape = (32, 32, 3)
    num_classes = 10

    inputs = layers.Input(shape=input_shape)

    # Block 1: Split and Convolution
    split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    conv_1x1 = Conv2D(32, (1, 1), activation='relu', padding='same')(split_channels[0])
    conv_3x3 = Conv2D(32, (3, 3), activation='relu', padding='same')(split_channels[1])
    conv_5x5 = Conv2D(32, (5, 5), activation='relu', padding='same')(split_channels[2])
    
    # Apply dropout to each feature map
    drop_1x1 = Dropout(0.3)(conv_1x1)
    drop_3x3 = Dropout(0.3)(conv_3x3)
    drop_5x5 = Dropout(0.3)(conv_5x5)

    # Concatenate the three feature maps
    block1_output = Concatenate(axis=-1)([drop_1x1, drop_3x3, drop_5x5])

    # Block 2: Inception-like structure
    # Branch 1
    branch1 = Conv2D(64, (1, 1), activation='relu', padding='same')(block1_output)
    
    # Branch 2
    branch2_1x1 = Conv2D(48, (1, 1), activation='relu', padding='same')(block1_output)
    branch2_3x3 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch2_1x1)

    # Branch 3
    branch3_1x1 = Conv2D(48, (1, 1), activation='relu', padding='same')(block1_output)
    branch3_5x5 = Conv2D(64, (5, 5), activation='relu', padding='same')(branch3_1x1)

    # Branch 4
    branch4_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(block1_output)
    branch4_1x1 = Conv2D(64, (1, 1), activation='relu', padding='same')(branch4_pool)

    # Concatenate all branches
    block2_output = Concatenate(axis=-1)([branch1, branch2_3x3, branch3_5x5, branch4_1x1])

    # Output Layer
    flatten = Flatten()(block2_output)
    outputs = Dense(num_classes, activation='softmax')(flatten)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# Example usage:
model = dl_model()
model.summary()