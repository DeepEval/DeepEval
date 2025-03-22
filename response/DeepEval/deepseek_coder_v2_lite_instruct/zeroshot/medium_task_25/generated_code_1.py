import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():
    # Define the input shape
    inputs = Input(shape=(32, 32, 3))

    # Path 1: Single 1x1 convolution
    path1 = Conv2D(64, (1, 1), activation='relu')(inputs)

    # Path 2: Average pooling followed by a 1x1 convolution
    path2 = AveragePooling2D(pool_size=(8, 8))(inputs)
    path2 = Conv2D(64, (1, 1), activation='relu')(path2)

    # Path 3: 1x1 convolution followed by two parallel 1x3 and 3x1 convolutions, then concatenate
    path3 = Conv2D(64, (1, 1), activation='relu')(inputs)
    path3_1 = Conv2D(32, (1, 3), padding='same', activation='relu')(path3)
    path3_2 = Conv2D(32, (3, 1), padding='same', activation='relu')(path3)
    path3 = Concatenate()([path3_1, path3_2])

    # Path 4: 1x1 convolution followed by a 3x3 convolution, then two parallel 1x3 and 3x1 convolutions, then concatenate
    path4 = Conv2D(64, (1, 1), activation='relu')(inputs)
    path4 = Conv2D(32, (3, 3), padding='same', activation='relu')(path4)
    path4_1 = Conv2D(32, (1, 3), padding='same', activation='relu')(path4)
    path4_2 = Conv2D(32, (3, 1), padding='same', activation='relu')(path4)
    path4 = Concatenate()([path4_1, path4_2])

    # Concatenate outputs of all paths
    combined_features = Concatenate()([path1, path2, path3, path4])

    # Flatten the combined features
    flattened_features = Flatten()(combined_features)

    # Output layer for classification
    outputs = Dense(10, activation='softmax')(flattened_features)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model