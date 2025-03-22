import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Dense, Concatenate, Add, Activation, BatchNormalization

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Convolutional layer
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)

    # Global average pooling
    gavgpool1 = GlobalAveragePooling2D()(conv1)
    gavgpool1_flat = Flatten()(gavgpool1)

    # Fully connected layers
    fc1 = Dense(64, activation='relu')(gavgpool1_flat)
    fc2 = Dense(128, activation='relu')(fc1)

    # Global max pooling
    gmaxpool1 = GlobalMaxPooling2D()(conv1)
    gmaxpool1_flat = Flatten()(gmaxpool1)

    # Fully connected layers
    fc3 = Dense(64, activation='relu')(gmaxpool1_flat)
    fc4 = Dense(128, activation='relu')(fc3)

    # Channel attention weights
    channel_weights = Add()([fc2, fc4])
    channel_weights = Activation('softmax')(channel_weights)

    # Element-wise multiplication with channel attention weights
    channel_attention = Concatenate()([conv1, channel_weights])

    # Average pooling
    avgpool1 = AveragePooling2D((2, 2))(channel_attention)

    # Max pooling
    maxpool1 = MaxPooling2D((2, 2))(channel_attention)

    # Concatenate spatial features
    spatial_features = Concatenate()([avgpool1, maxpool1])

    # Flatten and fully connected layer
    spatial_features = Flatten()(spatial_features)
    fc5 = Dense(256, activation='relu')(spatial_features)
    output = Dense(10, activation='softmax')(fc5)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output)

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model