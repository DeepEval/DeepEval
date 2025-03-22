import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, GlobalAveragePooling2D, Multiply, Concatenate
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def same_block(input_tensor):
        # Global Average Pooling
        gap = GlobalAveragePooling2D()(input_tensor)
        # Fully Connected Layers
        fc1 = Dense(128, activation='relu')(gap)
        fc2 = Dense(64, activation='relu')(fc1)
        # Reshape to match input shape
        reshaped = Dense(input_tensor.shape[1] * input_tensor.shape[2] * input_tensor.shape[3], activation='sigmoid')(fc2)
        reshaped = keras.backend.reshape(reshaped, (-1, input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))
        # Element-wise multiplication
        output_tensor = Multiply()([input_tensor, reshaped])
        return output_tensor

    # Apply same block to both branches
    branch1 = same_block(input_tensor=input_layer)
    branch2 = same_block(input_tensor=input_layer)

    # Concatenate outputs from both branches
    concatenated = Concatenate()([branch1, branch2])

    # Flatten the concatenated output
    flattened = Flatten()(concatenated)

    # Fully Connected Layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model