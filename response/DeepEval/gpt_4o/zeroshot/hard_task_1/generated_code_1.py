import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Multiply, Add, AveragePooling2D, Activation, Concatenate, Flatten
from tensorflow.keras.models import Model

def dl_model():
    # CIFAR-10 images are 32x32 with 3 color channels (RGB)
    input_shape = (32, 32, 3)
    num_classes = 10

    # Input layer
    inputs = Input(shape=input_shape)

    # Initial convolutional layer to adjust number of channels
    x = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu')(inputs)

    # Block 1: Channel Attention
    # Path 1
    path1 = GlobalAveragePooling2D()(x)
    path1 = Dense(units=3, activation='relu')(path1)
    path1 = Dense(units=3, activation='sigmoid')(path1)

    # Path 2
    path2 = GlobalMaxPooling2D()(x)
    path2 = Dense(units=3, activation='relu')(path2)
    path2 = Dense(units=3, activation='sigmoid')(path2)

    # Combine paths and apply channel attention
    channel_attention = Add()([path1, path2])
    channel_attention = Activation('sigmoid')(channel_attention)
    channel_attention = Multiply()([x, channel_attention])

    # Block 2: Spatial Attention
    avg_pool = AveragePooling2D(pool_size=(2, 2))(channel_attention)
    max_pool = GlobalMaxPooling2D()(channel_attention)
    max_pool = tf.expand_dims(tf.expand_dims(max_pool, 1), 1)
    max_pool = tf.tile(max_pool, [1, avg_pool.shape[1], avg_pool.shape[2], 1])
    
    spatial_features = Concatenate(axis=-1)([avg_pool, max_pool])
    spatial_features = Conv2D(filters=3, kernel_size=(1, 1), activation='sigmoid')(spatial_features)
    
    # Element-wise multiplication of spatial and channel attention features
    combined_features = Multiply()([channel_attention, spatial_features])

    # Additional branch to align channels
    aligned_features = Conv2D(filters=3, kernel_size=(1, 1), activation='relu')(combined_features)

    # Add to main path
    output = Add()([x, aligned_features])
    output = Activation('relu')(output)

    # Classification layer
    output = Flatten()(output)
    output = Dense(units=num_classes, activation='softmax')(output)

    # Construct the model
    model = Model(inputs=inputs, outputs=output)

    return model

# Example of creating the model
model = dl_model()
model.summary()