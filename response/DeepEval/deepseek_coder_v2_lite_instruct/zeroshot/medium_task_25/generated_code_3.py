import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Path 1: Single 1x1 convolution
    path1 = Conv2D(64, (1, 1), activation='relu')(input_layer)

    # Path 2: Average pooling followed by a 1x1 convolution
    path2 = AveragePooling2D(pool_size=(8, 8))(input_layer)
    path2 = Conv2D(64, (1, 1), activation='relu')(path2)

    # Path 3: 1x1 convolution followed by two parallel 1x3 and 3x1 convolutions
    path3_1x1 = Conv2D(64, (1, 1), activation='relu')(input_layer)
    path3_1x3 = Conv2D(64, (1, 3), padding='same', activation='relu')(path3_1x1)
    path3_3x1 = Conv2D(64, (3, 1), padding='same', activation='relu')(path3_1x1)
    path3 = Concatenate()([path3_1x3, path3_3x1])

    # Path 4: 1x1 convolution followed by a 3x3 convolution, then two parallel 1x3 and 3x1 convolutions
    path4_1x1 = Conv2D(64, (1, 1), activation='relu')(input_layer)
    path4_3x3 = Conv2D(64, (3, 3), padding='same', activation='relu')(path4_1x1)
    path4_1x3 = Conv2D(64, (1, 3), padding='same', activation='relu')(path4_3x3)
    path4_3x1 = Conv2D(64, (3, 1), padding='same', activation='relu')(path4_3x3)
    path4 = Concatenate()([path4_1x3, path4_3x1])

    # Concatenate the outputs of all paths
    merged_paths = Concatenate()([path1, path2, path3, path4])

    # Flatten the output and add a fully connected layer for classification
    flatten = Flatten()(merged_paths)
    output_layer = Dense(10, activation='softmax')(flatten)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
model = dl_model()
model.summary()