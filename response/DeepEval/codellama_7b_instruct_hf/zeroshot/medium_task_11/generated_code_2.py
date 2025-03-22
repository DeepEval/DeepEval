import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Add, Activation, Flatten

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the first convolutional layer
    x = Conv2D(64, (3, 3), activation='relu')(input_layer)

    # Define the first parallel path
    path1 = GlobalAveragePooling2D()(x)
    path1 = Dense(128, activation='relu')(path1)
    path1 = Dense(10, activation='softmax')(path1)

    # Define the second parallel path
    path2 = GlobalMaxPooling2D()(x)
    path2 = Dense(128, activation='relu')(path2)
    path2 = Dense(10, activation='softmax')(path2)

    # Define the attention weights
    attention_weights = Add()([path1, path2])
    attention_weights = Activation('sigmoid')(attention_weights)

    # Apply channel attention to the original features
    x = Multiply()([x, attention_weights])

    # Define the second convolutional layer
    x = Conv2D(128, (3, 3), activation='relu')(x)

    # Define the spatial features
    spatial_features = AveragePooling2D(pool_size=(2, 2))(x)
    spatial_features = MaxPooling2D(pool_size=(2, 2))(spatial_features)
    spatial_features = Flatten()(spatial_features)

    # Combine the channel and spatial features
    fused_features = Add()([spatial_features, x])

    # Define the final fully connected layer
    output_layer = Dense(10, activation='softmax')(fused_features)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model