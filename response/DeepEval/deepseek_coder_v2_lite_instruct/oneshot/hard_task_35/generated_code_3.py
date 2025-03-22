import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Multiply
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def same_block(input_tensor):
        # Global Average Pooling
        gap = GlobalAveragePooling2D()(input_tensor)
        # Fully connected layers
        fc1 = Dense(units=64, activation='relu')(gap)
        fc2 = Dense(units=32, activation='relu')(fc1)
        # Reshape to match input shape
        weights = Dense(units=input_tensor.shape[-1])(fc2)
        weights = keras.layers.Reshape((1, 1, input_tensor.shape[-1]))(weights)
        # Element-wise multiplication
        output_tensor = Multiply()([input_tensor, weights])
        return output_tensor

    # Branch 1
    branch1 = same_block(input_tensor=input_layer)
    # Branch 2
    branch2 = same_block(input_tensor=input_layer)

    # Concatenate outputs from both branches
    concatenated = Concatenate()([branch1, branch2])
    # Flatten the concatenated result
    flattened = Flatten()(concatenated)
    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flattened)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model