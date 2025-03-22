import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Multiply, Reshape
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def same_block(input_tensor):
        # Global Average Pooling
        gap = AveragePooling2D(pool_size=(8, 8))(input_tensor)
        # Fully connected layers
        fc1 = Dense(units=32, activation='relu')(gap)
        fc2 = Dense(units=32, activation='relu')(fc1)
        # Reshape to match input shape
        weights = Reshape((1, 1, 32))(fc2)
        # Element-wise multiplication
        output_tensor = Multiply()([input_tensor, weights])
        return output_tensor

    # Branch 1
    branch1 = same_block(input_tensor=input_layer)
    branch2 = same_block(input_tensor=input_layer)

    # Concatenate outputs from both branches
    concatenated = Concatenate()([branch1, branch2])
    # Flatten the result
    flattened = Flatten()(concatenated)
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model