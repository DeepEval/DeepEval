import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Add, Activation, Multiply, Flatten, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32x3

    # Initial convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Path 1: Global Average Pooling followed by Fully Connected Layers
    gap = GlobalAveragePooling2D()(conv)
    dense1_gap = Dense(units=128, activation='relu')(gap)
    dense2_gap = Dense(units=64, activation='relu')(dense1_gap)

    # Path 2: Global Max Pooling followed by Fully Connected Layers
    gmp = GlobalMaxPooling2D()(conv)
    dense1_gmp = Dense(units=128, activation='relu')(gmp)
    dense2_gmp = Dense(units=64, activation='relu')(dense1_gmp)

    # Combine the outputs of the two paths
    combined = Add()([dense2_gap, dense2_gmp])
    channel_attention_weights = Activation('sigmoid')(combined)

    # Apply channel attention weights to the original features
    channel_attended = Multiply()([conv, Activation('sigmoid')(channel_attention_weights)])

    # Separate Average and Max Pooling operations for spatial features
    spatial_avg = GlobalAveragePooling2D()(channel_attended)
    spatial_max = GlobalMaxPooling2D()(channel_attended)

    # Concatenate spatial features
    spatial_features = Concatenate()([spatial_avg, spatial_max])

    # Final fully connected layer for classification
    flatten = Flatten()(spatial_features)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model