import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Add, Activation, Multiply, Concatenate, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have a shape of 32x32x3

    # Initial convolutional layer
    conv_layer = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Path 1: Global Average Pooling followed by two Dense layers
    avg_pooling = GlobalAveragePooling2D()(conv_layer)
    dense_avg1 = Dense(units=128, activation='relu')(avg_pooling)
    dense_avg2 = Dense(units=32, activation='relu')(dense_avg1)

    # Path 2: Global Max Pooling followed by two Dense layers
    max_pooling = GlobalMaxPooling2D()(conv_layer)
    dense_max1 = Dense(units=128, activation='relu')(max_pooling)
    dense_max2 = Dense(units=32, activation='relu')(dense_max1)

    # Adding outputs from both paths
    channel_attention = Add()([dense_avg2, dense_max2])
    channel_attention = Activation('sigmoid')(channel_attention)

    # Applying channel attention to the original features
    attention_mul = Multiply()([conv_layer, tf.expand_dims(tf.expand_dims(channel_attention, axis=1), axis=2)])

    # Extracting spatial features with separate average and max pooling
    spatial_avg_pool = GlobalAveragePooling2D()(attention_mul)
    spatial_max_pool = GlobalMaxPooling2D()(attention_mul)

    # Concatenating spatial features along the channel dimension
    spatial_fused = Concatenate()([spatial_avg_pool, spatial_max_pool])

    # Flatten and Fully Connected Layer for final output
    flatten_layer = Flatten()(spatial_fused)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model