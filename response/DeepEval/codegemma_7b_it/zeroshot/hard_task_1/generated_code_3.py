import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Activation, Dropout, Flatten, Dense, Reshape, Multiply, concatenate, add, multiply

def ca_block(x, filters, ratio):
    # Path 1
    avg_pool = GlobalAveragePooling2D()(x)
    fc1 = Dense(filters // ratio, kernel_initializer='he_normal')(avg_pool)
    fc1 = Activation('relu')(fc1)
    fc2 = Dense(filters, kernel_initializer='he_normal')(fc1)
    fc2 = Activation('sigmoid')(fc2)

    # Path 2
    max_pool = GlobalMaxPooling2D()(x)
    fc1 = Dense(filters // ratio, kernel_initializer='he_normal')(max_pool)
    fc1 = Activation('relu')(fc1)
    fc2 = Dense(filters, kernel_initializer='he_normal')(fc1)
    fc2 = Activation('sigmoid')(fc2)

    # Channel Attention
    channel_attention = add([fc1, fc2])
    channel_attention = multiply([x, channel_attention])

    # Block 2
    avg_pool2 = AveragePooling2D()(x)
    max_pool2 = MaxPooling2D()(x)
    concat_pool2 = concatenate([avg_pool2, max_pool2])
    conv_pool2 = Conv2D(filters, (1, 1), padding='same')(concat_pool2)
    bn_pool2 = BatchNormalization()(conv_pool2)
    act_pool2 = Activation('sigmoid')(bn_pool2)

    # Spatial Attention
    spatial_attention = multiply([x, act_pool2])

    # Final Block
    concat_attention = add([channel_attention, spatial_attention])
    conv_attention = Conv2D(filters, (1, 1), padding='same')(concat_attention)
    bn_attention = BatchNormalization()(conv_attention)

    # Output
    output = add([bn_attention, x])
    output = Activation('relu')(output)

    return output

# Input layer
input_img = Input(shape=(32, 32, 3))

# Initial convolutional layer
x = Conv2D(64, (3, 3), padding='same')(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Block 1
x = ca_block(x, 64, 1)

# Block 2
x = ca_block(x, 128, 2)

# Block 3
x = ca_block(x, 256, 2)

# Block 4
x = ca_block(x, 512, 2)

# Global average pooling
avg_pool = GlobalAveragePooling2D()(x)

# Fully connected layer for classification
fc = Dense(10, kernel_initializer='he_normal')(avg_pool)
fc = Activation('softmax')(fc)

# Model
model = tf.keras.Model(inputs=input_img, outputs=fc)

# Return the model
return model