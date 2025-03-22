import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Flatten, Add, GlobalMaxPooling2D, Concatenate, Activation

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Define the initial convolutional layer
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)

    # Define the first path
    path1 = Conv2D(64, (3, 3), activation='relu')(conv1)
    path1 = GlobalAveragePooling2D()(path1)
    path1 = Dense(64, activation='relu')(path1)
    path1 = Dense(10, activation='softmax')(path1)

    # Define the second path
    path2 = Conv2D(64, (3, 3), activation='relu')(conv1)
    path2 = GlobalMaxPooling2D()(path2)
    path2 = Dense(64, activation='relu')(path2)
    path2 = Dense(10, activation='softmax')(path2)

    # Define the channel attention weights
    channel_weights = Add()([path1, path2])
    channel_weights = Activation('sigmoid')(channel_weights)

    # Define the fused feature map
    fused_features = Concatenate()([path1, path2])
    fused_features = Dense(64, activation='relu')(fused_features)
    fused_features = Dense(10, activation='softmax')(fused_features)

    # Define the final output
    output_layer = Add()([channel_weights * fused_features])

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model