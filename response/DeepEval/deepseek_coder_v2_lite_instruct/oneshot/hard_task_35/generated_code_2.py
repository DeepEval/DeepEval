import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Multiply, BatchNormalization
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def same_block(input_tensor):
        # Global Average Pooling
        gap = AveragePooling2D(pool_size=(input_tensor.shape[1:3]))(input_tensor)
        
        # Fully connected layers to produce weights
        fc1 = Dense(units=input_tensor.shape[-1], activation='relu')(gap)
        fc2 = Dense(units=input_tensor.shape[-1], activation='relu')(fc1)
        
        # Reshape weights to match input shape
        weights = Dense(units=input_tensor.shape[1] * input_tensor.shape[2] * input_tensor.shape[3], activation='sigmoid')(fc2)
        weights = keras.backend.reshape(weights, (input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))
        
        # Element-wise multiplication
        output_tensor = Multiply()([input_tensor, weights])
        
        return output_tensor

    # Two branches with the same block
    branch1 = same_block(input_tensor=input_layer)
    branch2 = same_block(input_tensor=input_layer)

    # Concatenate outputs of both branches
    concatenated = Concatenate()([branch1, branch2])

    # Flatten the concatenated output
    flattened = Flatten()(concatenated)

    # Fully connected layer for classification
    dense = Dense(units=10, activation='softmax')(flattened)

    model = Model(inputs=input_layer, outputs=dense)

    return model

# Example usage
model = dl_model()
model.summary()