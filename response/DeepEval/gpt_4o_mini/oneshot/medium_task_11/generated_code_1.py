import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Multiply, Flatten, Add, Activation, Concatenate
from keras.models import Model

def dl_model():
    # Input layer for CIFAR-10 images (32x32 pixels, 3 color channels)
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Path 1: Global Average Pooling followed by two fully connected layers
    avg_pool = GlobalAveragePooling2D()(conv)
    dense1_avg = Dense(units=128, activation='relu')(avg_pool)
    dense2_avg = Dense(units=64, activation='relu')(dense1_avg)

    # Path 2: Global Max Pooling followed by two fully connected layers
    max_pool = GlobalMaxPooling2D()(conv)
    dense1_max = Dense(units=128, activation='relu')(max_pool)
    dense2_max = Dense(units=64, activation='relu')(dense1_max)

    # Adding outputs from the two paths
    combined = Add()([dense2_avg, dense2_max])
    channel_attention = Activation('sigmoid')(combined)

    # Applying channel attention to original features
    attention_output = Multiply()([conv, channel_attention])

    # Average and Max Pooling to extract spatial features
    avg_pool_spatial = GlobalAveragePooling2D()(attention_output)
    max_pool_spatial = GlobalMaxPooling2D()(attention_output)

    # Concatenating spatial features along the channel dimension
    spatial_features = Concatenate()([avg_pool_spatial, max_pool_spatial])

    # Final dense layer
    flatten_layer = Flatten()(spatial_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Constructing the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model