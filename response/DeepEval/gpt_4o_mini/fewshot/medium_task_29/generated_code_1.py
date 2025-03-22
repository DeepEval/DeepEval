import keras
from keras.layers import Input, MaxPooling2D, Flatten, Concatenate, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels (RGB)

    # First pooling layer with 1x1 window
    maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    flatten1 = Flatten()(maxpool1)

    # Second pooling layer with 2x2 window
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    flatten2 = Flatten()(maxpool2)

    # Third pooling layer with 4x4 window
    maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    flatten3 = Flatten()(maxpool3)

    # Concatenate flattened outputs from all pooling layers
    combined_features = Concatenate()([flatten1, flatten2, flatten3])

    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(combined_features)
    dense2 = Dense(units=128, activation='relu')(dense1)

    # Output layer for classification (10 classes for CIFAR-10)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model