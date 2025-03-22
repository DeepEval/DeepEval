import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, GlobalAveragePooling2D, Multiply, Concatenate
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def shared_block(input_tensor):
        # Global Average Pooling
        gap = GlobalAveragePooling2D()(input_tensor)
        # Fully connected layers
        fc1 = Dense(128, activation='relu')(gap)
        fc2 = Dense(64, activation='relu')(fc1)
        # Reshape to match input shape
        reshaped = Dense(input_tensor.shape[1] * input_tensor.shape[2] * input_tensor.shape[3])(fc2)
        reshaped = Reshape((input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))(reshaped)
        # Element-wise multiplication with input
        output_tensor = Multiply()([input_tensor, reshaped])
        return output_tensor

    # Branch 1
    branch1 = shared_block(input_layer)
    # Branch 2
    branch2 = shared_block(input_layer)

    # Concatenate outputs from both branches
    concatenated = Concatenate()([branch1, branch2])
    # Flatten the concatenated output
    flattened = Flatten()(concatenated)
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model